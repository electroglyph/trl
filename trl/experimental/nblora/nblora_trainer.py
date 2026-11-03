# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset, load_from_disk
from torch import nn
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)

from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import get_config_model_id
from .nblora_config import NBLoraConfig


def merge_nblora_weight(weight, nblora_B, nblora_C, nblora_null, scaling):
    """
    Merge an NB-LoRA adapter into its backbone weight.

    Args:
        weight (`torch.Tensor`):
            Backbone weight `W` of shape `(m, n)`.
        nblora_B (`torch.Tensor`):
            Trainable factor `B` of shape `(m, r)`.
        nblora_C (`torch.Tensor`):
            Trainable factor `C` of shape `(r, k)`.
        nblora_null (`torch.Tensor`):
            Frozen null basis `Ns` of shape `(n, k)`.
        scaling (`float`):
            Adapter scaling factor (`alpha / rank`).

    Returns:
        `torch.Tensor`: `W + scaling * B C Ns^T`, in the dtype of `weight`.
    """
    delta = scaling * (nblora_B @ nblora_C @ nblora_null.to(device=nblora_B.device, dtype=nblora_B.dtype).t())
    return weight + delta.to(device=weight.device, dtype=weight.dtype)


class NBLoraLinear(nn.Module):
    """
    A frozen `nn.Linear` wrapped with a trainable Null-Basis LoRA adapter.

    The forward pass computes `x W^T + (alpha/r) ((x Ns) C^T) B^T` where `W` is the frozen backbone weight, `Ns` is
    the frozen orthonormal null basis, and `B`/`C` are the only trainable parameters. At initialization `B` is zero,
    so the adapter contributes exactly nothing.
    """

    def __init__(self, base: nn.Linear, null_basis: torch.Tensor, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        # Shared with the original module (no copy): backbone state_dict keys are unchanged.
        self.weight = base.weight
        self.bias = base.bias
        self.register_buffer("nblora_null", null_basis.to(device=base.weight.device, dtype=torch.float32))
        self.nblora_B = nn.Parameter(
            torch.zeros(base.weight.shape[0], rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.nblora_C = nn.Parameter(
            torch.empty(rank, null_basis.shape[1], device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.nblora_C, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.nblora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        h = self.nblora_dropout(x) @ self.nblora_null.to(x.dtype)
        adapter = (h @ self.nblora_C.t()) @ self.nblora_B.t()
        return base + self.scaling * adapter

    def merged_weight(self) -> torch.Tensor:
        """
        Return the backbone weight with the adapter merged in.

        Returns:
            `torch.Tensor`: `W + (alpha/r) B C Ns^T`, in the backbone dtype.
        """
        return merge_nblora_weight(self.weight, self.nblora_B, self.nblora_C, self.nblora_null, self.scaling)


def _collect_gram_matrices(model, tokenizer, target_names, texts, max_length):
    """
    Run anchor prompts through the model and accumulate per-site Gram matrices `G += H^T H` in fp32 on CPU.

    Modules that receive the same input tensor object (e.g. Q/K/V, or MLP gate/up) share one site: site membership is
    detected from runtime tensor identity on a probe pass, never from module names. Only one representative module per
    site accumulates, so shared inputs are counted exactly once.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            Model in its training dtype/device. Left in its original training mode afterwards.
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer used to tokenize the anchor texts.
        target_names (`list[str]`):
            Dotted names of the target `nn.Linear` modules.
        texts (`list[str]`):
            Anchor prompts.
        max_length (`int`):
            Maximum tokenized length per anchor prompt.

    Returns:
        `tuple[dict, dict]`: per-site Gram matrices (`{site: G}`, fp32 CPU) and site membership (`{target: site}`).
    """
    modules = dict(model.named_modules())
    probe_inputs = {}

    def probe_hook(name):
        def hook(module, args):
            probe_inputs[name] = args[0]

        return hook

    was_training = model.training
    model.eval()
    try:
        handles = [modules[name].register_forward_pre_hook(probe_hook(name)) for name in target_names]
        try:
            batch = tokenizer(texts[0], truncation=True, max_length=max_length, return_tensors="pt")
            model(input_ids=batch["input_ids"].to(next(model.parameters()).device))
        finally:
            for handle in handles:
                handle.remove()
        # Group by runtime tensor identity; kept references in probe_inputs keep ids unique.
        rep_of_key = {}
        site_of = {}
        for name in target_names:
            site_of[name] = rep_of_key.setdefault(id(probe_inputs[name]), name)
        reps = sorted(set(site_of.values()))
        in_dims = {rep: probe_inputs[rep].shape[-1] for rep in reps}
        grams = {rep: torch.zeros(dim, dim, dtype=torch.float32) for rep, dim in in_dims.items()}

        def accum_hook(rep):
            def hook(module, args):
                h = args[0].detach().to(torch.float32).cpu().reshape(-1, args[0].shape[-1])
                grams[rep] += h.t() @ h

            return hook

        handles = [modules[rep].register_forward_pre_hook(accum_hook(rep)) for rep in reps]
        try:
            with torch.no_grad():
                device = next(model.parameters()).device
                for text in texts:
                    batch = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
                    model(input_ids=batch["input_ids"].to(device))
        finally:
            for handle in handles:
                handle.remove()
    finally:
        model.train(was_training)
    return grams, site_of


def _build_null_bases(grams, energy_threshold):
    """
    Eigendecompose each Gram matrix and keep the tail eigenvectors outside the top spectral energy as null bases.

    Args:
        grams (`dict[str, torch.Tensor]`):
            Per-site Gram matrices `{site: G}`.
        energy_threshold (`float`):
            Spectral energy `eps` covered by the frozen top eigenvectors.

    Returns:
        `dict[str, torch.Tensor]`: frozen orthonormal null bases `{site: Ns}` (`n×k`). Sites with `k == 0` or `k ==
        n` (or empty Gram) are skipped with a warning and keep their plain linear layer.
    """
    null_bases = {}
    for site, G in grams.items():
        n = G.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(G)
        total = eigenvalues.sum().item()
        if total <= 0:
            warnings.warn(f"Skipping NB-LoRA adapter for site '{site}': Gram matrix is empty.", stacklevel=2)
            continue
        protected = int(torch.searchsorted((eigenvalues.flip(0).cumsum(0) / total), energy_threshold).item()) + 1
        protected = min(max(protected, 1), n)
        k = n - protected
        if k <= 0 or k >= n:
            warnings.warn(
                f"Skipping NB-LoRA adapter for site '{site}': null-space dimension k={k} (n={n}).", stacklevel=2
            )
            continue
        null_bases[site] = eigenvectors[:, :k].contiguous()
    return null_bases


def _resolve_anchor_texts(anchor_dataset, num_examples):
    """
    Resolve the anchor dataset to a list of prompt strings.

    Args:
        anchor_dataset (`Dataset` or `list[str]` or `str`):
            A [`~datasets.Dataset`] (uses the `"text"` column, else the first string column), a list of prompt
            strings, or a dataset identifier/path (loaded with `split="train"`, or from disk for a local path).
        num_examples (`int`):
            Number of anchor prompts to keep.

    Returns:
        `list[str]`:
            The first `num_examples` anchor prompts.
    """
    if isinstance(anchor_dataset, str):
        dataset = (
            load_from_disk(anchor_dataset)
            if os.path.isdir(anchor_dataset)
            else load_dataset(anchor_dataset, split="train")
        )
    elif isinstance(anchor_dataset, Dataset):
        dataset = anchor_dataset
    elif isinstance(anchor_dataset, list):
        dataset = None
        texts = anchor_dataset
    else:
        raise TypeError(
            "`nblora_anchor_dataset` must be a `Dataset`, a list of strings, or a dataset identifier/path, got "
            f"{type(anchor_dataset).__name__}."
        )
    if dataset is not None:
        if len(dataset) == 0:
            raise ValueError("`nblora_anchor_dataset` is empty.")
        columns = dataset.column_names
        if "text" in columns:
            column = "text"
        else:
            column = next((c for c in columns if isinstance(dataset[0][c], str)), None)
            if column is None:
                raise ValueError(
                    "`nblora_anchor_dataset` has no `text` column and no string column to fall back to "
                    f"(columns: {columns})."
                )
        texts = dataset.select(range(min(num_examples, len(dataset))))[column]
    if len(texts) == 0:
        raise ValueError("`nblora_anchor_dataset` is empty.")
    if any(not isinstance(t, str) for t in texts):
        raise ValueError("`nblora_anchor_dataset` must contain strings.")
    return list(texts[:num_examples])


class NBLoraTrainer(SFTTrainer):
    """
    Trainer for Null-Basis LoRA (NB-LoRA) fine-tuning.

    NB-LoRA (https://huggingface.co/papers/2609.25618) constrains the update of each target `nn.Linear` to the null
    space of its input activations on reasoning rollouts, so post-RL reasoning survives downstream SFT. The trainer
    reuses the [`SFTTrainer`] tokenization/packing/loss; it only wraps the model: anchor Gram matrices are collected
    with forward hooks (streamed `G += H^T H` in fp32 on CPU, one prompt at a time), tail eigenvectors become frozen
    null bases, and each target linear becomes an [`NBLoraLinear`][trl.experimental.nblora.nblora_trainer] whose only
    trainable parameters are `B` and `C`.

    Only `nn.Linear` modules are supported, and a single device is assumed for null-basis collection. To persist a
    fully merged checkpoint, call [`merge_and_unload`][trl.experimental.nblora.nblora_trainer] before `save_model`;
    otherwise `save_model` stores the frozen backbone together with the small adapter state and cached null bases.

    Example:

    ```python
    >>> from trl.experimental.nblora import NBLoraConfig, NBLoraTrainer
    >>> training_args = NBLoraConfig(
    ...     output_dir="nblora",
    ...     nblora_target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
    ...     nblora_anchor_dataset="gsm8k",
    ... )
    >>> trainer = NBLoraTrainer(model="Qwen/Qwen2.5-1.5B", args=training_args, train_dataset=dataset)
    >>> trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`]):
            Model to be trained. A string is interpreted as a model id (or local path) and loaded with the keyword
            arguments in `args.model_init_kwargs`.
        args ([`experimental.nblora.NBLoraConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used (which still requires
            `nblora_target_modules` to be set before training).
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
        train_dataset ([`~datasets.Dataset`]):
            Dataset to use for training. Same formats as [`SFTTrainer`].
        eval_dataset ([`~datasets.Dataset`], [`~datasets.DatasetDict`] or `dict[str, Dataset]`, *optional*):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, it is loaded from the model's name.
        compute_loss_func (`Callable`, *optional*):
            Deprecated, as in [`SFTTrainer`]: subclass and override `compute_loss` instead.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step.
        quantization_config ([`~transformers.BitsAndBytesConfig`], *optional*):
            Quantization configuration used when loading the model from a model identifier.
        peft_config ([`~peft.PeftConfig`], *optional*):
            Not supported: NB-LoRA manages its own adapter. Passing one raises a `ValueError`.
        formatting_func (`Callable[[dict], str]`, *optional*):
            Formatting function applied to the dataset before tokenization.
    """

    _tag_names = ["trl", "nblora"]

    def __init__(
        self,
        model: "str | PreTrainedModel",
        args: NBLoraConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple | None = (None, None),
        optimizer_cls_and_kwargs: tuple | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        quantization_config=None,
        peft_config=None,
        formatting_func: Callable | None = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = NBLoraConfig(f"{model_name.split('/')[-1]}-NBLora")
        elif not isinstance(args, NBLoraConfig):
            raise TypeError(
                f"`args` must be an `NBLoraConfig`, got {type(args).__name__}. Convert an `SFTConfig` with "
                "`NBLoraConfig(**args.to_dict(), nblora_target_modules=[...])`."
            )
        if peft_config is not None:
            raise ValueError("`peft_config` is not supported by `NBLoraTrainer`: NB-LoRA manages its own adapter.")
        if not args.nblora_target_modules:
            raise ValueError(
                "`nblora_target_modules` is required (no default): list the `nn.Linear` suffixes to adapt."
            )
        if args.nblora_rank < 1:
            raise ValueError(f"`nblora_rank` must be >= 1, got {args.nblora_rank}.")
        if args.nblora_alpha < 0:
            raise ValueError(f"`nblora_alpha` must be >= 0, got {args.nblora_alpha}.")
        if not 0.0 <= args.nblora_dropout < 1.0:
            raise ValueError(f"`nblora_dropout` must be in [0, 1), got {args.nblora_dropout}.")
        if not 0.0 < args.nblora_energy_threshold < 1.0:
            raise ValueError(f"`nblora_energy_threshold` must be in (0, 1), got {args.nblora_energy_threshold}.")
        if args.nblora_num_anchor_examples < 1:
            raise ValueError(f"`nblora_num_anchor_examples` must be >= 1, got {args.nblora_num_anchor_examples}.")
        if args.nblora_anchor_max_length < 1:
            raise ValueError(f"`nblora_anchor_max_length` must be >= 1, got {args.nblora_anchor_max_length}.")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            quantization_config=quantization_config,
            peft_config=None,
            formatting_func=formatting_func,
        )

        target_names = self._target_linear_names()
        self.nblora_anchor_texts: list[str] = []
        null_bases, site_of = self._get_null_bases(target_names)
        self.nblora_site_of = site_of
        self.nblora_null_bases = null_bases
        self._apply_nblora_adapters(target_names)

    def _target_linear_names(self) -> list[str]:
        """
        Return the dotted names of the target `nn.Linear` modules, preserving model order.

        Returns:
            `list[str]`: dotted names matching one of `nblora_target_modules` on a component boundary.
        """
        suffixes = self.args.nblora_target_modules
        modules = {name: module for name, module in self.model.named_modules() if isinstance(module, nn.Linear)}
        for suffix in suffixes:
            if not any(name == suffix or name.endswith("." + suffix) for name in modules):
                raise ValueError(
                    f"`nblora_target_modules` entry {suffix!r} matched no `nn.Linear` module. "
                    f"Model has {len(modules)} `nn.Linear` modules."
                )
        matched = [name for name in modules if any(name == s or name.endswith("." + s) for s in suffixes)]
        # The SFT loss projects through `lm_head.weight` directly (fused LM head) without calling the module, so an
        # adapter there would receive no gradients and crash training: reject it like SFTTrainer does for PEFT.
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None:
            for name in matched:
                if modules[name] is output_embeddings:
                    raise ValueError(
                        f"`nblora_target_modules` entry targeting {name!r} is not supported: the SFT loss reads "
                        "`lm_head.weight` directly (fused LM head) and never calls the module, so the adapter "
                        "would receive no gradients. Target hidden-layer linears instead."
                    )
        return list(dict.fromkeys(matched))

    def _get_null_bases(self, target_names: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
        """
        Load cached null bases or build them from the anchor dataset.

        Args:
            target_names (`list[str]`):
                Dotted names of the target `nn.Linear` modules.

        Returns:
            `tuple[dict, dict]`: null bases (`{site: Ns}`, fp32 CPU) and site membership (`{target: site}`).
        """
        cache_path = self.args.nblora_null_cache_path
        if cache_path is not None and os.path.isfile(cache_path):
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            valid = (
                isinstance(cached, dict)
                and set(cached.get("site_of", {})) == set(target_names)
                and cached.get("energy_threshold") == self.args.nblora_energy_threshold
            )
            if valid:
                # Sites skipped at build time (k == 0 or k == n) have no entry and stay skipped. They are tied to the
                # model checkpoint the cache was built from, like the bases themselves.
                modules = dict(self.model.named_modules())
                for name, site in cached["site_of"].items():
                    Ns = cached["null_bases"].get(site)
                    if Ns is not None and Ns.shape[0] != modules[name].in_features:
                        valid = False
                        break
            if valid:
                return cached["null_bases"], cached["site_of"]
            warnings.warn(
                f"Ignoring NB-LoRA cache at '{cache_path}': it does not match the current model/targets/energy. "
                "Rebuilding.",
                stacklevel=2,
            )
        if self.args.nblora_anchor_dataset is None:
            raise ValueError(
                "`nblora_anchor_dataset` is required (no usable `nblora_null_cache_path` found): pass the reasoning "
                "prompts used to build the null bases."
            )
        texts = _resolve_anchor_texts(self.args.nblora_anchor_dataset, self.args.nblora_num_anchor_examples)
        self.nblora_anchor_texts = texts
        grams, site_of = _collect_gram_matrices(
            self.model, self._tokenizer, target_names, texts, self.args.nblora_anchor_max_length
        )
        null_bases = _build_null_bases(grams, self.args.nblora_energy_threshold)
        if cache_path is not None:
            parent = os.path.dirname(os.path.abspath(cache_path))
            os.makedirs(parent, exist_ok=True)
            torch.save(
                {
                    "energy_threshold": self.args.nblora_energy_threshold,
                    "site_of": site_of,
                    "null_bases": null_bases,
                },
                cache_path,
            )
        return null_bases, site_of

    def _apply_nblora_adapters(self, target_names: list[str]) -> None:
        """
        Replace target linears with [`NBLoraLinear`][trl.experimental.nblora.nblora_trainer] and freeze everything
        except the `B`/`C` factors.

        Args:
            target_names (`list[str]`):
                Dotted names of the target `nn.Linear` modules.
        """
        modules = dict(self.model.named_modules())
        wrapped = 0
        for name in target_names:
            site = self.nblora_site_of[name]
            if site not in self.nblora_null_bases:
                continue
            base = modules[name]
            adapter = NBLoraLinear(
                base,
                self.nblora_null_bases[site],
                rank=self.args.nblora_rank,
                alpha=self.args.nblora_alpha,
                dropout=self.args.nblora_dropout,
            )
            parent_path, _, attr = name.rpartition(".")
            parent = modules[parent_path] if parent_path else self.model
            setattr(parent, attr, adapter)
            wrapped += 1
        if wrapped == 0:
            warnings.warn("No NB-LoRA adapter was attached: every site was skipped.", stacklevel=2)
        for param_name, param in self.model.named_parameters():
            param.requires_grad = "nblora_B" in param_name or "nblora_C" in param_name

    def compute_constraint_residual(self, texts: list[str] | None = None, max_examples: int = 8) -> float:
        """
        Measure how well the adapter preserves the null-space constraint on anchor-like inputs.

        Runs `max_examples` prompts through the model in evaluation mode and returns the worst-case ratio `||H ΔW^T||
        / ||H W^T||` over wrapped modules and probe prompts. By construction (`H Ns ≈ 0`) this stays near zero before
        and after training.

        Args:
            texts (`list[str]`, *optional*):
                Prompts to probe with. Defaults to the first anchor prompts used at build time.
            max_examples (`int`, *optional*, defaults to `8`):
                Number of prompts to probe with.

        Returns:
            `float`: worst-case relative constraint residual (`0.0` when no adapter is attached).
        """
        wrapped = [(name, module) for name, module in self.model.named_modules() if isinstance(module, NBLoraLinear)]
        if not wrapped:
            return 0.0
        if texts is None:
            if len(self.nblora_anchor_texts) == 0:
                raise ValueError(
                    "`texts` is required here: the trainer was built from `nblora_null_cache_path`, so no anchor "
                    "prompts are stored."
                )
            texts = self.nblora_anchor_texts
        probe_texts = list(texts[:max_examples])
        if len(probe_texts) == 0:
            raise ValueError("`texts` is empty.")
        captured = {}

        def capture_hook(name):
            def hook(module, args):
                captured[name] = args[0].detach()

            return hook

        was_training = self.model.training
        self.model.eval()
        worst = 0.0
        try:
            handles = [module.register_forward_pre_hook(capture_hook(name)) for name, module in wrapped]
            try:
                with torch.no_grad():
                    device = next(self.model.parameters()).device
                    for text in probe_texts:
                        batch = self._tokenizer(
                            text, truncation=True, max_length=self.args.nblora_anchor_max_length, return_tensors="pt"
                        )
                        self.model(input_ids=batch["input_ids"].to(device))
                        for name, module in wrapped:
                            h = captured[name].to(torch.float32)
                            base_out = h @ module.weight.detach().to(torch.float32).t()
                            delta_out = (
                                (
                                    (h @ module.nblora_null.to(torch.float32))
                                    @ module.nblora_C.detach().to(torch.float32).t()
                                )
                                @ module.nblora_B.detach().to(torch.float32).t()
                            ) * module.scaling
                            denom = base_out.norm().item()
                            if denom == 0:
                                continue
                            worst = max(worst, (delta_out.norm().item() / denom))
            finally:
                for handle in handles:
                    handle.remove()
        finally:
            self.model.train(was_training)
        return worst

    def merge_and_unload(self):
        """
        Merge every NB-LoRA adapter into its backbone weight, in place.

        Each [`NBLoraLinear`][trl.experimental.nblora.nblora_trainer] is replaced by a plain `nn.Linear` with weight `W
        + (alpha/r) B C Ns^T` (frozen). Call before `save_model` to persist a fully merged checkpoint.

        Returns:
            [`~transformers.PreTrainedModel`]: the model with adapters merged.
        """
        modules = dict(self.model.named_modules())
        for name, module in list(modules.items()):
            if not isinstance(module, NBLoraLinear):
                continue
            merged = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            merged.weight = nn.Parameter(module.merged_weight(), requires_grad=False)
            if module.bias is not None:
                merged.bias = module.bias
                merged.bias.requires_grad_(False)
            merged.to(device=module.weight.device, dtype=module.weight.dtype)
            parent_path, _, attr = name.rpartition(".")
            parent = modules[parent_path] if parent_path else self.model
            setattr(parent, attr, merged)
        self.nblora_null_bases = {}
        return self.model
