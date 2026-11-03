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

import pytest
import torch
from datasets import Dataset
from torch import nn
from transformers import AutoModelForCausalLM

from trl.experimental.nblora import NBLoraConfig, NBLoraTrainer
from trl.experimental.nblora.nblora_trainer import (
    _build_null_bases,
    _resolve_anchor_texts,
    merge_nblora_weight,
)
from trl.trainer import SFTConfig

from ..testing_utils import TrlTestCase, require_peft


TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_MOE_MODEL = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
QKV_TARGETS = ["q_proj", "k_proj", "v_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]


def _reasoning_texts(n=12):
    return [
        f"Question: What is {i} plus {i + 1}? Let's think step by step. First add the numbers, then answer: {2 * i + 1}."
        for i in range(n)
    ]


def _decaying_gram(n=16):
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    spectrum = torch.tensor([2.0**-i for i in range(n)])
    return Q @ torch.diag(spectrum) @ Q.t()


def _sft_texts(n=8):
    return [f"Review {i}: this short sentence is used for supervised fine-tuning." for i in range(n)]


def _make_trainer(tmp_dir, anchor_texts=None, targets=QKV_TARGETS, cache_path=None, no_anchor=False, **overrides):
    if no_anchor:
        anchor = None
        num_anchor = 512
    else:
        if anchor_texts is None:
            anchor_texts = _reasoning_texts()
        anchor = Dataset.from_dict({"text": anchor_texts})
        num_anchor = len(anchor_texts)
    config_kwargs = {
        "output_dir": tmp_dir,
        "report_to": "none",
        "save_strategy": "no",
        "max_length": 64,
        "gradient_checkpointing": False,
        # The tiny test model (hidden size 8) has a flat activation spectrum, so the paper-default energy of 0.99
        # leaves no null space. A lower energy exercises the same machinery; the 0.99 default gets its own test.
        "nblora_energy_threshold": 0.5,
        "nblora_target_modules": targets,
        "nblora_anchor_dataset": anchor,
        "nblora_num_anchor_examples": num_anchor,
        "nblora_null_cache_path": cache_path,
    }
    config_kwargs.update(overrides)
    args = NBLoraConfig(**config_kwargs)
    return NBLoraTrainer(
        model=TINY_MODEL,
        args=args,
        train_dataset=Dataset.from_dict({"text": _sft_texts()}),
    )


class TestNBLoraConfig:
    def test_defaults_match_paper_table_8(self):
        args = NBLoraConfig("dummy")
        assert args.nblora_rank == 16
        assert args.nblora_alpha == 32
        assert args.nblora_dropout == 0.05
        assert args.nblora_energy_threshold == 0.99
        assert args.nblora_num_anchor_examples == 512
        assert args.nblora_anchor_max_length == 1024

    def test_no_model_specific_defaults(self):
        args = NBLoraConfig("dummy")
        assert args.nblora_target_modules is None
        assert args.nblora_anchor_dataset is None
        assert args.nblora_null_cache_path is None

    def test_is_sft_config(self):
        assert isinstance(NBLoraConfig("dummy"), SFTConfig)


class TestPureHelpers:
    def test_merge_weight_matches_manual_update(self):
        torch.manual_seed(0)
        weight = torch.randn(6, 8)
        B = torch.randn(6, 4)
        C = torch.randn(4, 5)
        Ns, _ = torch.linalg.qr(torch.randn(8, 5))
        merged = merge_nblora_weight(weight, B, C, Ns, scaling=2.0)
        expected = weight + 2.0 * (B @ C @ Ns.t())
        torch.testing.assert_close(merged, expected)

    def test_merge_scaling_enters_linearly(self):
        torch.manual_seed(0)
        weight = torch.randn(6, 8)
        B = torch.randn(6, 4)
        C = torch.randn(4, 3)
        Ns, _ = torch.linalg.qr(torch.randn(8, 3))
        small = merge_nblora_weight(weight, B, C, Ns, scaling=1.0)
        big = merge_nblora_weight(weight, B, C, Ns, scaling=2.0)
        torch.testing.assert_close(big - weight, 2 * (small - weight))

    def test_build_null_bases_keeps_tail(self):
        G = _decaying_gram()
        (null_bases,) = _build_null_bases({"site": G}, 0.99).values()
        n, k = null_bases.shape
        assert n == 16 and 0 < k < n
        # Orthonormal: Ns^T Ns = I
        torch.testing.assert_close(null_bases.t() @ null_bases, torch.eye(k), rtol=1e-4, atol=1e-5)
        # Tail: kept energy is below 1 - eps only by the protected top part
        kept_energy = (G * (null_bases @ null_bases.t())).sum() / torch.trace(G)
        assert kept_energy.item() < 0.05

    def test_build_null_bases_k_grows_when_energy_shrinks(self):
        G = _decaying_gram()
        k_loose = _build_null_bases({"site": G}, 0.90)["site"].shape[1]
        k_strict = _build_null_bases({"site": G}, 0.99)["site"].shape[1]
        assert k_loose >= k_strict > 0

    def test_build_null_bases_skips_empty_gram(self):
        with pytest.warns(UserWarning, match="Gram matrix is empty"):
            assert _build_null_bases({"site": torch.zeros(8, 8)}, 0.99) == {}

    def test_resolve_anchor_texts_from_list(self):
        assert _resolve_anchor_texts(["a", "b", "c"], 2) == ["a", "b"]

    def test_resolve_anchor_texts_from_dataset_text_column(self):
        dataset = Dataset.from_dict({"text": ["x", "y"], "other": [0, 1]})
        assert _resolve_anchor_texts(dataset, 10) == ["x", "y"]

    def test_resolve_anchor_texts_falls_back_to_first_string_column(self):
        dataset = Dataset.from_dict({"question": ["q0", "q1"], "label": [0, 1]})
        assert _resolve_anchor_texts(dataset, 10) == ["q0", "q1"]

    def test_resolve_anchor_texts_rejects_bad_inputs(self):
        with pytest.raises(TypeError):
            _resolve_anchor_texts(123, 4)
        with pytest.raises(ValueError, match="empty"):
            _resolve_anchor_texts([], 4)
        with pytest.raises(ValueError, match="empty"):
            _resolve_anchor_texts(Dataset.from_dict({"text": []}), 4)
        with pytest.raises(ValueError, match="strings"):
            _resolve_anchor_texts(["ok", 42], 2)
        with pytest.raises(ValueError, match="no `text` column"):
            _resolve_anchor_texts(Dataset.from_dict({"a": [1, 2]}), 2)


class TestNBLoraTrainerInit(TrlTestCase):
    def test_target_modules_required(self):
        args = NBLoraConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="nblora_target_modules"):
            NBLoraTrainer(model=TINY_MODEL, args=args, train_dataset=Dataset.from_dict({"text": _sft_texts()}))

    def test_unknown_target_module(self):
        with pytest.raises(ValueError, match="matched no `nn.Linear` module"):
            _make_trainer(self.tmp_dir, targets=["no_such_proj"])

    def test_lm_head_target_rejected(self):
        # The SFT loss reads lm_head.weight directly (fused LM head), so the adapter would get no gradients.
        with pytest.raises(ValueError, match="lm_head"):
            _make_trainer(self.tmp_dir, targets=["lm_head"])

    def test_sft_config_rejected_with_hint(self):
        args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(TypeError, match="NBLoraConfig"):
            NBLoraTrainer(model=TINY_MODEL, args=args, train_dataset=Dataset.from_dict({"text": _sft_texts()}))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"nblora_energy_threshold": 1.0},
            {"nblora_energy_threshold": 0.0},
            {"nblora_rank": 0},
            {"nblora_alpha": -1},
            {"nblora_dropout": 1.0},
            {"nblora_num_anchor_examples": 0},
            {"nblora_anchor_max_length": 0},
        ],
    )
    def test_invalid_hyperparams_rejected(self, kwargs):
        with pytest.raises(ValueError, match="nblora_"):
            _make_trainer(self.tmp_dir, **kwargs)

    @require_peft
    def test_peft_config_rejected(self):
        from peft import LoraConfig

        args = NBLoraConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            nblora_target_modules=QKV_TARGETS,
            nblora_anchor_dataset=Dataset.from_dict({"text": _reasoning_texts()}),
        )
        with pytest.raises(ValueError, match="peft_config"):
            NBLoraTrainer(
                model=TINY_MODEL,
                args=args,
                train_dataset=Dataset.from_dict({"text": _sft_texts()}),
                peft_config=LoraConfig(),
            )

    def test_missing_anchor_rejected_without_cache(self):
        args = NBLoraConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            nblora_target_modules=QKV_TARGETS,
            nblora_energy_threshold=0.5,
        )
        with pytest.raises(ValueError, match="nblora_anchor_dataset"):
            NBLoraTrainer(model=TINY_MODEL, args=args, train_dataset=Dataset.from_dict({"text": _sft_texts()}))


class TestNBLoraStructure(TrlTestCase):
    def test_default_energy_gracefully_skips_flat_spectrum(self):
        # Paper-default energy 0.99 leaves no null space on the tiny model (hidden size 8, flat spectrum): every
        # site is skipped with a warning instead of crashing.
        with pytest.warns(UserWarning, match="No NB-LoRA adapter was attached"):
            trainer = _make_trainer(self.tmp_dir, nblora_energy_threshold=0.99)
        assert trainer.nblora_null_bases == {}
        assert trainer.compute_constraint_residual() == 0.0
        assert not any(p.requires_grad for p in trainer.model.parameters())
        merged = trainer.merge_and_unload()  # no-op, must not fail
        assert not any("nblora_" in key for key in merged.state_dict())

    def test_qkv_share_one_site_per_layer(self):
        trainer = _make_trainer(self.tmp_dir, targets=QKV_TARGETS)
        layers = {name.rsplit(".q_proj", 1)[0] for name in trainer.nblora_site_of if name.endswith(".q_proj")}
        assert len(layers) >= 1
        for layer in layers:
            sites = {trainer.nblora_site_of[f"{layer}.{proj}"] for proj in ("q_proj", "k_proj", "v_proj")}
            assert len(sites) == 1
        # One site per layer, fewer sites than target modules
        assert len(trainer.nblora_null_bases) == len(layers) < len(trainer.nblora_site_of)

    def test_mlp_gate_up_share_down_separate(self):
        trainer = _make_trainer(self.tmp_dir, targets=MLP_TARGETS)
        layers = {name.rsplit(".gate_proj", 1)[0] for name in trainer.nblora_site_of if name.endswith(".gate_proj")}
        assert len(layers) >= 1
        for layer in layers:
            gate_site = trainer.nblora_site_of[f"{layer}.gate_proj"]
            assert trainer.nblora_site_of[f"{layer}.up_proj"] == gate_site
            assert trainer.nblora_site_of[f"{layer}.down_proj"] != gate_site
        assert len(trainer.nblora_null_bases) == 2 * len(layers)

    def test_site_detection_ignores_names(self):
        # Grouping must not depend on suffix cleverness: full dotted names group identically.
        full = [f"model.layers.0.self_attn.{proj}" for proj in ("q_proj", "k_proj", "v_proj")]
        trainer = _make_trainer(self.tmp_dir, targets=full)
        assert {trainer.nblora_site_of[name] for name in full} == set(trainer.nblora_null_bases)
        assert len(trainer.nblora_null_bases) == 1

    def test_null_bases_orthonormal(self):
        trainer = _make_trainer(self.tmp_dir, targets=QKV_TARGETS + MLP_TARGETS)
        assert len(trainer.nblora_null_bases) > 0
        for site, Ns in trainer.nblora_null_bases.items():
            n, k = Ns.shape
            assert 0 < k < n, site
            torch.testing.assert_close(Ns.t() @ Ns, torch.eye(k), rtol=1e-4, atol=1e-5)

    def test_only_adapter_params_trainable(self):
        trainer = _make_trainer(self.tmp_dir)
        trainable = {name for name, param in trainer.model.named_parameters() if param.requires_grad}
        assert len(trainable) > 0
        assert all("nblora_B" in name or "nblora_C" in name for name in trainable)
        n_BC = sum(1 for _, param in trainer.model.named_parameters() if param.requires_grad)
        n_wrapped = sum(1 for module in trainer.model.modules() if module.__class__.__name__ == "NBLoraLinear")
        assert n_BC == 2 * n_wrapped

    def test_trainable_param_count(self):
        trainer = _make_trainer(self.tmp_dir, targets=QKV_TARGETS + MLP_TARGETS)
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        expected = 0
        vanilla = 0
        for module in trainer.model.modules():
            if module.__class__.__name__ != "NBLoraLinear":
                continue
            m, n = module.weight.shape
            k = module.nblora_null.shape[1]
            r = trainer.args.nblora_rank
            expected += m * r + r * k
            vanilla += m * r + r * n
        assert trainable == expected
        assert trainable < vanilla  # mr + rk < mr + rn since k < n

    def test_delta_zero_at_init(self):
        trainer = _make_trainer(self.tmp_dir)
        for module in trainer.model.modules():
            if module.__class__.__name__ != "NBLoraLinear":
                continue
            torch.testing.assert_close(module.merged_weight(), module.weight)

    def test_nblora_linear_forward_matches_base_at_init(self):
        from trl.experimental.nblora.nblora_trainer import NBLoraLinear

        torch.manual_seed(0)
        base = nn.Linear(8, 6)
        Ns, _ = torch.linalg.qr(torch.randn(8, 5))
        adapter = NBLoraLinear(base, Ns, rank=4, alpha=32, dropout=0.0).eval()
        x = torch.randn(2, 3, 8)
        torch.testing.assert_close(adapter(x), base(x))

    def test_constraint_residual_zero_at_init(self):
        # B is initialized to zero, so the adapter contributes exactly nothing: the residual is exactly 0.0.
        trainer = _make_trainer(self.tmp_dir)
        assert trainer.compute_constraint_residual() == 0.0

    def test_constraint_residual_needs_texts_when_built_from_cache(self):
        cache_path = os.path.join(self.tmp_dir, "Ns_cache", "null.pt")
        _make_trainer(self.tmp_dir, cache_path=cache_path)
        cached = _make_trainer(self.tmp_dir, no_anchor=True, cache_path=cache_path)
        assert cached.compute_constraint_residual(_reasoning_texts(4)) < 1e-3
        with pytest.raises(ValueError, match="texts"):
            cached.compute_constraint_residual()


class TestNBLoraMerge(TrlTestCase):
    def _eval_logits(self, trainer, texts):
        device = next(trainer.model.parameters()).device
        batch = trainer.processing_class(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        trainer.model.eval()
        with torch.no_grad():
            return trainer.model(input_ids=batch["input_ids"]).logits.cpu()

    def test_merge_matches_adapter_forward(self):
        trainer = _make_trainer(self.tmp_dir)
        texts = _sft_texts(2)
        before = self._eval_logits(trainer, texts)
        trainer.merge_and_unload()
        assert not any("nblora_" in name for name, _ in trainer.model.named_parameters())
        assert not any("nblora_" in key for key in trainer.model.state_dict())
        after = self._eval_logits(trainer, texts)
        torch.testing.assert_close(after, before)

    def test_merge_fp64_matches_adapter_forward(self):
        # High-precision merge path (disabled by default): same merge, computed in float64.
        trainer = _make_trainer(self.tmp_dir)
        texts = _sft_texts(2)
        before = self._eval_logits(trainer, texts)
        trainer.merge_and_unload(dtype=torch.float64)
        assert not any("nblora_" in key for key in trainer.model.state_dict())
        after = self._eval_logits(trainer, texts)
        torch.testing.assert_close(after, before)

    def test_merge_then_save_and_reload(self):
        trainer = _make_trainer(self.tmp_dir)
        before = self._eval_logits(trainer, _sft_texts(2))
        trainer.merge_and_unload()
        merged_dir = os.path.join(self.tmp_dir, "merged")
        trainer.save_model(merged_dir)
        reloaded = AutoModelForCausalLM.from_pretrained(merged_dir)
        batch = trainer.processing_class(
            _sft_texts(2), padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            after = reloaded(input_ids=batch["input_ids"]).logits
        torch.testing.assert_close(after, before)


class TestNBLoraCache(TrlTestCase):
    def test_cache_roundtrip(self):
        cache_path = os.path.join(self.tmp_dir, "Ns_cache", "null.pt")
        first = _make_trainer(self.tmp_dir, cache_path=cache_path)
        assert os.path.isfile(cache_path)
        second = _make_trainer(self.tmp_dir, no_anchor=True, cache_path=cache_path)
        assert set(second.nblora_null_bases) == set(first.nblora_null_bases)
        assert second.nblora_site_of == first.nblora_site_of
        for site in first.nblora_null_bases:
            torch.testing.assert_close(second.nblora_null_bases[site], first.nblora_null_bases[site])

    def test_cache_with_disk_anchor_dataset(self):
        anchor = Dataset.from_dict({"text": _reasoning_texts()})
        anchor_dir = os.path.join(self.tmp_dir, "anchor_ds")
        anchor.save_to_disk(anchor_dir)
        cache_path = os.path.join(self.tmp_dir, "null.pt")
        args = NBLoraConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            save_strategy="no",
            max_length=64,
            gradient_checkpointing=False,
            nblora_energy_threshold=0.5,
            nblora_target_modules=QKV_TARGETS,
            nblora_anchor_dataset=anchor_dir,
            nblora_num_anchor_examples=12,
            nblora_null_cache_path=cache_path,
        )
        trainer = NBLoraTrainer(model=TINY_MODEL, args=args, train_dataset=Dataset.from_dict({"text": _sft_texts()}))
        assert len(trainer.nblora_null_bases) > 0

    def test_stale_cache_rebuilds_with_warning(self):
        cache_path = os.path.join(self.tmp_dir, "null.pt")
        _make_trainer(self.tmp_dir, targets=QKV_TARGETS, cache_path=cache_path)
        with pytest.warns(UserWarning, match="Ignoring NB-LoRA cache"):
            rebuilt = _make_trainer(self.tmp_dir, targets=MLP_TARGETS, cache_path=cache_path)
        assert len(rebuilt.nblora_null_bases) > 0
        # Cache now matches the new targets, so a third build reuses it silently.
        reused = _make_trainer(self.tmp_dir, no_anchor=True, targets=MLP_TARGETS, cache_path=cache_path)
        for site in rebuilt.nblora_null_bases:
            torch.testing.assert_close(reused.nblora_null_bases[site], rebuilt.nblora_null_bases[site])


class TestNBLoraTrain(TrlTestCase):
    def test_train_updates_only_adapter(self):
        trainer = _make_trainer(
            self.tmp_dir, learning_rate=1e-4, max_steps=2, per_device_train_batch_size=2, logging_steps=1
        )
        before_trainable = {
            name: param.detach().clone() for name, param in trainer.model.named_parameters() if param.requires_grad
        }
        before_frozen = {
            name: param.detach().clone() for name, param in trainer.model.named_parameters() if not param.requires_grad
        }
        trainer.train()
        loss = trainer.state.log_history[-1]["train_loss"]
        assert loss is not None and math.isfinite(loss)
        for name, old in before_trainable.items():
            assert not torch.equal(old, trainer.model.get_parameter(name)), f"{name} did not change"
        for name, old in before_frozen.items():
            assert torch.equal(old, trainer.model.get_parameter(name)), f"frozen {name} changed"
        # The null-space constraint survives optimization. The 1e-3 acceptance bar from the spec assumes the
        # paper-regime energy of 0.99 (1% tail energy); these tests run at energy 0.5 on the tiny model, where the
        # tail holds more energy, so a looser bound applies. A broken constraint (full-rank update) would read O(1).
        assert trainer.compute_constraint_residual() < 1e-2

    def test_train_with_gradient_checkpointing(self):
        trainer = _make_trainer(
            self.tmp_dir,
            learning_rate=1e-4,
            max_steps=1,
            per_device_train_batch_size=2,
            logging_steps=1,
            gradient_checkpointing=True,
        )
        trainer.train()
        loss = trainer.state.log_history[-1]["train_loss"]
        assert loss is not None and math.isfinite(loss)

    def test_anchor_max_length_truncates(self):
        trainer = _make_trainer(self.tmp_dir, nblora_anchor_max_length=8)
        assert len(trainer.nblora_null_bases) > 0
        for Ns in trainer.nblora_null_bases.values():
            torch.testing.assert_close(Ns.t() @ Ns, torch.eye(Ns.shape[1]), rtol=1e-4, atol=1e-5)

    def test_num_anchor_examples_subset(self):
        trainer = _make_trainer(self.tmp_dir, nblora_num_anchor_examples=4)
        assert len(trainer.nblora_anchor_texts) == 4
        assert len(trainer.nblora_null_bases) > 0


class TestNBLoraMoeModel(TrlTestCase):
    def _make_moe_trainer(self, **overrides):
        config_kwargs = {
            "output_dir": self.tmp_dir,
            "report_to": "none",
            "save_strategy": "no",
            "max_length": 64,
            "gradient_checkpointing": False,
            "nblora_energy_threshold": 0.5,
            "nblora_target_modules": QKV_TARGETS,
            "nblora_anchor_dataset": Dataset.from_dict({"text": _reasoning_texts()}),
            "nblora_num_anchor_examples": 12,
        }
        config_kwargs.update(overrides)
        return NBLoraTrainer(
            model=TINY_MOE_MODEL,
            args=NBLoraConfig(**config_kwargs),
            train_dataset=Dataset.from_dict({"text": _sft_texts()}),
        )

    def test_moe_attention_targets_share_sites(self):
        # MoE experts are fused 3D-weight modules (not nn.Linear), so only the dense attention linears match.
        trainer = self._make_moe_trainer()
        assert len(trainer.nblora_null_bases) == 2  # one shared q/k/v site per layer
        for site, Ns in trainer.nblora_null_bases.items():
            n, k = Ns.shape
            assert 0 < k < n, site
            torch.testing.assert_close(Ns.t() @ Ns, torch.eye(k), rtol=1e-4, atol=1e-5)

    def test_moe_fused_expert_suffixes_rejected(self):
        with pytest.raises(ValueError, match="matched no `nn.Linear` module"):
            self._make_moe_trainer(nblora_target_modules=MLP_TARGETS)

    def test_moe_train_updates_only_adapter(self):
        # Two steps: with B initialized to zero, C receives no gradient on the first step (as in standard LoRA).
        trainer = self._make_moe_trainer(learning_rate=1e-4, max_steps=2, per_device_train_batch_size=2)
        trainable_before = {
            name: param.detach().clone() for name, param in trainer.model.named_parameters() if param.requires_grad
        }
        assert len(trainable_before) > 0
        trainer.train()
        loss = trainer.state.log_history[-1]["train_loss"]
        assert loss is not None and math.isfinite(loss)
        for name, old in trainable_before.items():
            assert not torch.equal(old, trainer.model.get_parameter(name)), f"{name} did not change"
        assert trainer.compute_constraint_residual() < 1e-2
