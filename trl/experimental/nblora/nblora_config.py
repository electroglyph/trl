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

from dataclasses import dataclass, field
from typing import Any

from ...trainer.sft_config import SFTConfig


@dataclass
class NBLoraConfig(SFTConfig):
    r"""
    Configuration class for the [`experimental.nblora.NBLoraTrainer`].

    [`experimental.nblora.NBLoraConfig`] inherits every parameter from [`SFTConfig`]; it only adds the parameters that
    control the Null-Basis LoRA (NB-LoRA) adapter from the paper [Reasoning-Preserving Fine-Tuning of Post-RL LLMs
    with Null-Basis LoRA](https://huggingface.co/papers/2609.25618). For each target `nn.Linear` layer with weight `W
    in R^{m×n}`, NB-LoRA collects input hidden states `H` from reasoning rollouts, forms the Gram matrix `G = H^T H`,
    keeps the tail eigenvectors outside the top-`nblora_energy_threshold` spectral energy as a frozen orthonormal null
    basis `Ns in R^{n×k}`, and trains only `B in R^{m×r}` and `C in R^{r×k}` with `ΔW = (alpha/r) B C Ns^T`, so `H ΔW^T
    = 0` by construction. The `(alpha/r)` scaling and the dropout placement follow standard LoRA convention; the paper
    lists `alpha=32` and dropout `0.05` (Table 8) without stating how they enter the update.

    Parameters:
        > Parameters that control the NB-LoRA adapter

        nblora_rank (`int`, *optional*, defaults to `16`):
            Rank `r` of the trainable factors `B` (`m×r`) and `C` (`r×k`). Paper value (Table 8).
        nblora_alpha (`int`, *optional*, defaults to `32`):
            LoRA-style scaling; the adapter branch is multiplied by `nblora_alpha / nblora_rank`. Paper value
            (Table 8); the `alpha/r` application itself is an engineering choice following standard LoRA convention.
        nblora_dropout (`float`, *optional*, defaults to `0.05`):
            Dropout applied to the adapter-branch input. Paper value (Table 8); placement follows LoRA convention.
        nblora_energy_threshold (`float`, *optional*, defaults to `0.99`):
            Spectral energy `eps` defining the approximate null space: the top eigenvectors covering `eps` of the Gram
            energy stay in the frozen backbone, the tail becomes the null basis. Paper value (Table 8).
        nblora_target_modules (`list[str]`, *optional*):
            **Required, no default.** Suffixes matched against the trailing component of each `nn.Linear` submodule
            name (e.g. `["q_proj", "k_proj", "v_proj", "w1", "w3", "w2"]`). Every entry must match at least one
            `nn.Linear` module. No model-specific default and no exclusion list: pass exactly the modules to adapt.
            The output head (`lm_head`) cannot be targeted: the SFT loss reads `lm_head.weight` directly (fused LM
            head) and never calls the module, so the adapter would receive no gradients.
        nblora_anchor_dataset (`Dataset` or `list[str]` or `str`, *optional*):
            Reasoning prompts `Dr` used to build the null bases (paper anchor corpora: MATH-lighteval, GSM8K). Either
            a [`~datasets.Dataset`] with a `"text"` column, a list of prompt strings, or a dataset identifier/path
            (loaded with `split="train"`). Not needed when `nblora_null_cache_path` points to an existing cache.
        nblora_num_anchor_examples (`int`, *optional*, defaults to `512`):
            Number of anchor prompts used for Gram-matrix collection. Paper value (Table 8).
        nblora_anchor_max_length (`int`, *optional*, defaults to `1024`):
            Maximum tokenized length of each anchor prompt. Paper value for commonsense adaptation; use `4096` for
            tool-use adaptation (Table 8).
        nblora_null_cache_path (`str`, *optional*):
            Path of a file used to save/load the null bases (`{energy_threshold, site_of, null_bases}`) to avoid
            recomputation. Parent directories are created on save.
    """

    nblora_rank: int = field(
        default=16,
        metadata={"help": "Rank r of the trainable NB-LoRA factors B (m×r) and C (r×k). Paper value (Table 8)."},
    )
    nblora_alpha: int = field(
        default=32,
        metadata={
            "help": "LoRA-style scaling; the adapter branch is multiplied by nblora_alpha / nblora_rank. Paper value "
            "(Table 8); the alpha/r application follows standard LoRA convention."
        },
    )
    nblora_dropout: float = field(
        default=0.05,
        metadata={
            "help": "Dropout applied to the adapter-branch input. Paper value (Table 8); placement follows LoRA "
            "convention."
        },
    )
    nblora_energy_threshold: float = field(
        default=0.99,
        metadata={
            "help": "Spectral energy eps defining the approximate null space. The top eigenvectors covering eps of "
            "the Gram energy stay frozen; the tail becomes the null basis. Paper value (Table 8)."
        },
    )
    nblora_target_modules: list[str] | None = field(
        default=None,
        metadata={
            "help": "Required, no default. Suffixes matched against the trailing component of each nn.Linear "
            "submodule name. Every entry must match at least one nn.Linear module. The output head (lm_head) "
            "cannot be targeted: the SFT loss reads lm_head.weight directly and never calls the module."
        },
    )
    nblora_anchor_dataset: Any = field(
        default=None,
        metadata={
            "help": "Reasoning prompts used to build the null bases: a Dataset with a 'text' column, a list of "
            "prompt strings, or a dataset identifier/path (loaded with split='train'). Not needed when "
            "nblora_null_cache_path points to an existing cache."
        },
    )
    nblora_num_anchor_examples: int = field(
        default=512,
        metadata={"help": "Number of anchor prompts used for Gram-matrix collection. Paper value (Table 8)."},
    )
    nblora_anchor_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum tokenized length of each anchor prompt. Paper value for commonsense adaptation; use "
            "4096 for tool-use adaptation (Table 8)."
        },
    )
    nblora_null_cache_path: str | None = field(
        default=None,
        metadata={
            "help": "Path of a file used to save/load the null bases ({energy_threshold, site_of, null_bases}) to "
            "avoid recomputation. Parent directories are created on save."
        },
    )
