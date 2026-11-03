# NB-LoRA

In the paper [Reasoning-Preserving Fine-Tuning of Post-RL LLMs with Null-Basis LoRA](https://huggingface.co/papers/2609.25618), the authors propose Null-Basis LoRA (NB-LoRA): for each target `nn.Linear` layer, input hidden states from reasoning rollouts are collected into a Gram matrix, whose tail eigenvectors (outside the top spectral energy) form a frozen orthonormal null basis. Only two small factors are trained, with the update constrained to that null space, so post-RL reasoning survives downstream SFT.

To use NB-LoRA, you can use the [`experimental.nblora.NBLoraTrainer`] class in `trl.experimental.nblora`.

## Usage

```python
from trl.experimental.nblora import NBLoraConfig, NBLoraTrainer

training_args = NBLoraConfig(
    output_dir="nblora",
    nblora_target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],  # required, no default
    nblora_anchor_dataset="gsm8k",  # reasoning prompts for the null bases; paper default count is 512
    nblora_rank=16,  # paper, Table 8
    nblora_alpha=32,  # paper, Table 8; applied as alpha/rank per LoRA convention
)
trainer = NBLoraTrainer(
    model="Qwen/Qwen2.5-1.5B",
    args=training_args,
    train_dataset=...,
)
trainer.train()
```

`nblora_target_modules` is always user-supplied: suffixes are matched against each `nn.Linear` submodule name. Layers sharing the same input activations (e.g. Q/K/V) share one null basis, detected at runtime. The output head (`lm_head`) cannot be targeted: the SFT loss reads `lm_head.weight` directly (fused LM head) and never calls the module, so the adapter would receive no gradients. Call `trainer.merge_and_unload()` before `save_model` to persist a fully merged checkpoint; otherwise the frozen backbone is saved together with the small adapter state and cached null bases.

## NBLoraTrainer

[[autodoc]] experimental.nblora.NBLoraTrainer
    - train
    - save_model
    - push_to_hub

## NBLoraConfig

[[autodoc]] experimental.nblora.NBLoraConfig
