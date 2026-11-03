# Plan: NB-LoRA in TRL (paper 2609.25618v1) — tested on LFM3-1.7B / RTX 3090 Ti 24GB

Paper: https://arxiv.org/abs/2609.25618v1
(HF mirror for code/docs links: https://huggingface.co/papers/2609.25618)

## 0. Goal

Add experimental support for **Null-Basis LoRA (NB-LoRA)** from
https://huggingface.co/papers/2609.25618
(`Reasoning-Preserving Fine-Tuning of Post-RL LLMs with Null-Basis LoRA`)
so post-RL reasoning survives downstream SFT.

Method in one paragraph: for each target `nn.Linear` (`W in R^{m×n}`), collect
input hidden states `H in R^{N×n}` from reasoning rollouts, form
`G = H^T H`, eigendecompose, keep tail eigenvectors outside top-`eps`
spectral energy as frozen orthonormal null basis `Ns in R^{n×k}`, and train
only `B in R^{m×r}`, `C in R^{r×k}` with `ΔW = B C Ns^T`, so
`H ΔW^T = 0` by construction. Trainable params per layer `mr + rk`
(< vanilla LoRA `mr + rn`). Standard optimizer, no per-step projection.

Non-goal: no change to stable `trl/trainer/*`. This lives in
`trl/experimental/` per `AGENTS.md` (main vs experimental code).

## 1. Target model (`/home/anon/LLM/LFM3-1.7B/`)

* `config.json`: 24 layers, `hidden_size=2048`, `intermediate_size=7680`,
  custom `Lfm3ForCausalLM` (`trust_remote_code=True`), `tie_word_embeddings=true`,
  `bf16`, GQA (16 Q / 4 KV heads), hybrid `conv / full_attention / sliding_attention`.
* Linear modules in `modeling_lfm3.py`:
  `attention.{q_proj,k_proj,v_proj,o_proj,gate_proj}`,
  `mlp.{w1,w3,w2}` (w1/w3 = up, w2 = down), `conv.{in_proj,out_proj}`, `lm_head`.
* Implications (examples only — implementation stays model-agnostic):
  * `target_modules` is **always user-supplied, no model-specific default**.
    Example for this model: `["q_proj","k_proj","v_proj","w1","w3","w2"]`
    (paper §4.3: Q/K/V alone underfits; MLP carries adaptation; paper's adapted set is exactly Q,K,V,up,down — `o_proj` is never adapted there, so omitting it matches the paper (see §9 fix H1; prior wording "`o_proj` optional" wrongly implied the paper evaluated it)).
    Exclude `lm_head`: targeting it raises an error — the SFT loss reads
    `lm_head.weight` directly (fused LM head) and never calls the module, so the
    adapter would receive no gradients. The code has no model-specific default
    and no silent exclusion list.
  * Layers sharing the same input `H` share one `G`/`Ns` (paper §3.2) —
    detect by shared input tensor site at runtime, not by name
    (e.g. `q/k/v` share one site, `w1`/`w3` share another on LFM3).
  * Custom-code model: all test scripts must pass
    `trust_remote_code=True` (on BOTH model and tokenizer loads — a bare
    `AutoTokenizer.from_pretrained` without it triggers the interactive
    custom-code prompt, observed live 2026-11-03) + local path `/home/anon/LLM/LFM3-1.7B/`.

## 2. API design (keep it lean, no registries/factories)

New package `trl/experimental/nblora/`:

* `nblora_config.py`: `NBLoraConfig(SFTConfig)` + only NB-specific fields:
  * `nblora_rank (int=16)`: `r`. Paper default (Table 8).
  * `nblora_alpha (int=32)`: LoRA-style scaling applied as `(alpha/r)` on the
    adapter branch, following standard LoRA convention (Hu et al. 2022) — the
    paper's equations (Eq.4, Alg.1 line 10/12) write the update as plain
    `B C Ns^T` with no scaling factor, while Table 8 lists `LoRA scaling α=32`
    without stating how it enters. So `alpha=32` is the paper value but the
    `α/r` application is an engineering choice (default scale factor 2);
    see §7 fix G1.
  * `nblora_dropout (float=0.05)`: dropout on the adapter branch, following LoRA
    convention (applied to the adapter input). Value is the paper default
    (Table 8 lists `LoRA dropout 0.05` with no placement stated); placement is
    an engineering choice, see §7 fix G1.
  * `nblora_energy_threshold (float=0.99)`: `eps` spectral energy defining the
    approximate null space (paper §3.3/§3.4, Table 8 "Protected spectral energy 99%").
  * `nblora_target_modules (list[str])`: **required, user-supplied, no default**.
    Generic suffix match against any `nn.Linear` submodule name
    (e.g. `q_proj`, `w1`, `dense`, `fc1` — works for Llama/Qwen/LFM3/any arch).
    LFM3 example (not a default): `["q_proj","k_proj","v_proj","w1","w3","w2"]`
    mirroring the paper's "Query, key, value, up-, and down-projections" (Table 8).
    Error clearly if a name matches nothing. Targeting the output head
    (`lm_head`) is rejected with an error: the SFT loss reads `lm_head.weight`
    directly (fused LM head) and never calls the module, so the adapter would
    receive no gradients. This is a functional guard, not a model-specific
    exclusion list.
  * `nblora_anchor_dataset`: `Dataset` or `list[str]` or `str` path/id of reasoning prompts `Dr`.
    Paper anchor corpora: MATH-lighteval (3B model), GSM8K (1.5B model) (Table 8).
  * `nblora_num_anchor_examples (int=512)`: paper value (Table 8).
    Smoke tests may use 64–128: Fig.5b shows retention stable 64–512 and target
    accuracy stable 32–512, degrading mainly at 32 (see §7, fix F4).
  * `nblora_anchor_max_length (int=1024)`: paper max sequence length is
    task-dependent (1024 commonsense / 4096 tool-use, Table 8); default 1024,
    override to 4096 for tool-use adaptation.
  * `nblora_null_cache_path (str|None)`: save/load `{energy_threshold, site_of,
    null_bases}` (`{site: Ns}` plus the site membership and energy used to build
    it) to avoid recompute. Cache is validated against current targets/energy/
    input dims; a stale cache rebuilds with a warning.
* `nblora_trainer.py`: `NBLoraTrainer(SFTTrainer)` — reuse SFT
  tokenization/packing/loss; only override model wrapping. Follows
  `experimental/gmpo/` precedent (subclass, override minimal surface).
* No separate `null_basis.py` in v1: small helpers inline in trainer file
  (`merge_nblora_weight`, `NBLoraLinear`, `_collect_gram_matrices`,
  `_build_null_bases`, `_resolve_anchor_texts`). The two Gram/null-basis
  functions are ~100 lines combined.
  Split out only if reused elsewhere.
* Merge on save: `W_merged = W + (alpha/r) B C Ns^T` (include the scale factor
  used in forward so merged logits match adapter logits exactly),
  standard `save_pretrained`
  (frozen backbone + small adapter state + cached `Ns`).

Docstrings follow repo format (backtick types, `*optional*`, `defaults to X`);
paper link uses `https://huggingface.co/papers/<id>` form.

## 3. Implementation steps

1. **Scaffold**: `trl/experimental/nblora/__init__.py`,
   `nblora_config.py`, `nblora_trainer.py`. Do NOT touch
   `trl/experimental/__init__.py` — it contains only the experimental-warning
   boilerplate and no per-trainer re-exports (verified); consumers import
   `from trl.experimental.nblora import ...` exactly like
   `from trl.experimental.gmpo import ...`. Copy license header
   from `gmpo`/`sdft`.
2. **Hook-based `G` collection** (no full-`H` materialization, critical on 3090):
   * Register forward hooks on target-linears' inputs during anchor forward passes.
   * Accumulate in fp32 on CPU: `G_l += H_batch^T H_batch` (per site, not per proj).
   * `G` for `n=2048` is 2048²×4B ≈ 16MB; for `w2` input `n=7680` ≈ 236MB fp32 —
     keep on CPU, one layer at a time. Use `torch.einsum` / `H.T @ H` per micro-batch.
   * Run model in `eval`, `bf16`, `torch.no_grad`, micro-batch 1–2, `max_length 1024`.
3. **Null basis**: `eigh(G)` → eigenvalues asc/desc, cumulative energy,
   cut at `eps` (default 0.99), stack tail vectors as `Ns` (`n×k`).
   Guard: if `k==0` or `k==n`, skip layer (log warning, fall back to no adapter).
   Orthonormality check `||Ns^T Ns − I||` in test.
4. **Adapter module**: replace each target `nn.Linear` with wrapper holding:
   * frozen `weight` (original), frozen buffer `Ns`, trainable `B (m×r)` and
     `C (r×k)`; forward `W x + (alpha/r) B(C(Ns^T x))` with paper values
     `r=16`, `alpha=32` (Table 8; the `α/r` factor itself follows LoRA convention
     and is not in the paper's equations — see §7 fix G1). Dropout 0.05 on the
     adapter branch per Table 8 (placement per LoRA convention).
   * Init (engineering choice, paper is silent — see §7 fix F2): `B=zeros`,
     `C=kaiming`, mirroring standard LoRA (A kaiming / B zeros) so that
     `ΔW=0` at step 0. Here `C Ns^T` plays the role of LoRA's `A`.
   * Only `B`,`C` require grad; backbone frozen. Support `gradient_checkpointing`
     + `bf16` passthrough. No `hasattr`/`getattr` branching.
5. **Train**: standard `SFTTrainer` SFT loop on adaptation set `Da`.
   Verify `H ΔW^T ≈ 0` on a held-out anchor batch before/after (norm ratio test).
6. **`paper_index.md`**: add subsection under the `## Parameter-Efficient
   Fine-Tuning (PEFT)` section (which hosts LoRA/DoRA entries; an SFT section
   also exists) linking
   `https://huggingface.co/papers/2609.25618` with `NBLoraConfig` snippet.
   Required by `AGENTS.md` for paper implementations.
7. **Docs**: `docs/source/nblora.md` minimal (config + 20-line example),
   following the `gmpo.md` short-name pattern for experimental trainers, plus a
   `local: nblora` entry in `docs/source/_toctree.yml` (where `gmpo` is
   registered today).
8. **Tests**: `tests/experimental/test_nblora.py`:
   * Real tiny model (`trl-internal-testing` tiny, e.g. `tiny-Qwen2`-style),
     real tokenizer/config/dataset — no `MagicMock`/`SimpleNamespace`/`object.__new__`.
   * Public API only: build trainer, check (a) `Ns` orthonormal + constraint residual
     small, (b) train 1–2 steps loss decreases, (c) trainable-param count
     `mr+rk < mr+rn`, (d) merge produces same logits as adapter forward.
   * Gate optional-dependency/hardware tests with `require_*` from
     `tests/testing_utils.py`.

## 4. 3090 test plan (model `/home/anon/LLM/LFM3-1.7B/`)

Environment (verified 2026-11-03, re-verified 2026-11-04): `~/.venv` python 3.12.13,
torch 2.13+cu130, transformers 5.18, peft 0.21.2, datasets 5.1, pytest 9.1.1,
`trl` resolving to this repo (`/home/anon/trl/trl/__init__.py`).
`nvidia-smi` reports RTX 3090 Ti with 23028 MiB. LFM3-1.7B loads bf16 on cuda
(`trust_remote_code=True` required — plain load prompts/errors on the custom
`lfm3` type), 2-prompt forward ok, 3.47GB peak (re-measured 3.468GB
`max_memory_allocated`, 2026-11-04) — plenty of headroom.
Use `~/.venv/bin/python` for all runs (`/usr/bin/python3` has no torch —
verified `import torch` fails there).

Model bf16 ≈ 3.5GB weights. Use
`per_device_batch_size=1–2`, `gradient_accumulation_steps=8–16`,
`gradient_checkpointing=True`, `bf16=True`, `max_length=1024`, `packing=False`
for smoke test.

* **Smoke A — null-basis build**: 64 anchor prompts (GSM8K/MATH subset, whichever
  is available offline; else 64 synthetic math CoT prompts in repo), verify per-site
  null-space share is large (paper Table 1 on Qwen2.5-3B at 99% energy: per-layer
  12.3–67.3% of `n`, module averages 29.2–45.1% — see §7 fix F3; expect the same
  order of magnitude on LFM3, not exact numbers),
  save cache to `/tmp/opencode/Ns_cache/`.
* **Smoke B — adaptation**: 200–500 commonsense/tool-use SFT rows, rank 16,
  alpha 32, dropout 0.05, lr 1e-4 (paper Table 8 commonsense; 2e-4 for tool-use),
  3 epochs, batch 16 where memory allows else grad-accum equivalent.
  Checks: loss drops, peak CUDA < 20GB, steps/sec logged,
  `H ΔW^T` residual ~0 on anchor batch.
* **Retention check** (cheap): before/after accuracy on 50–100 GSM8K/MATH-style
  prompts (greedy decode, exact-match) + target-task accuracy. Expect pattern from
  paper Table 2: vanilla LoRA collapses reasoning, NB-LoRA holds near pre-SFT.
  Full 9-task (8 commonsense + tool-use) + 5-benchmark sweep (paper §4) is out of
  scope for 3090 v1.
* **Ablations (only if time)**: `eps ∈ {0.90, 0.99}`, `anchors ∈ {64, 512}`,
  `target_modules` ∈ {Q/K/V-only, MLP-up/down-only, Q/K/V+MLP-up, all-five}
  (the paper Fig.5a grid) — paper Fig.5 predicts Q/K/V-only underfits, `eps=0.75`
  collapses retention, 32 anchors degrades (see §7 fix F4).

## 5. Acceptance criteria

* [ ] `NBLoraTrainer` trains LFM3-1.7B on 3090 without OOM (bf16 + ckpt).
* [ ] Constraint holds: `||H B C Ns^T|| / ||H W^T|| < 1e-3` on anchor sample.
* [ ] Target SFT loss decreases; reasoning anchor accuracy ≥ pre-SFT − noise,
  matching vanilla-LoRA-beats-collapse expectation.
* [ ] `paper_index.md` updated; new tests pass; no `trl/trainer/*` touched.

## 6. Risks / open questions

* LFM3-1.7B here is likely **not** the paper's post-RL checkpoint — anchor =
  reasoning eval set is a proxy; retention numbers test mechanics, not paper claims.
* `w2` (`7680→2048`) null basis is the memory hot spot — CPU fp32 `G` + streaming
  accumulation is mandatory; skip `w2` first if OOM.
* Conv `in_proj/out_proj` + `gate_proj` excluded in v1; revisit only if adaptation
  underfits.
* `eigh` on the 7680² `G` is the compute hot spot: symmetric eigendecomposition is
  O(n³) ≈ 6e11 flops at n=7680, i.e. tens of seconds to minutes on CPU — do once,
  cache to disk (`nblora_null_cache_path`). The 2048² sites take seconds.

## 7. Audit log (review vs paper + repo, 2026-11-03)

Method: re-read `null.md` against the arXiv HTML of 2609.25618v1
(extracted to `/tmp/opencode/nb_paper.txt`, ~91k chars) and against the repo
(`trl/experimental/__init__.py`, `docs/source/` listing, LFM3 modeling file).
Every change below quotes its evidence.

* **F1 — added arXiv link at top.** Previously only the HF mirror
  `https://huggingface.co/papers/2609.25618` was present. The HF link is kept
  because `AGENTS.md` ("Links to papers") mandates the
  `https://huggingface.co/papers/<id>` form for code/docs references; the arXiv
  URL is added per explicit user request. Both IDs share the same suffix system,
  so they denote the same paper.
* **F2 — hyperparameters corrected to paper Table 8 (Appendix D.1).**
  The draft used SFTConfig-flavored guesses (`rank=8`, `lr=2e-5`, 1 epoch,
  no alpha/dropout). Table 8 ("Training and null-space construction
  hyperparameters for NB-LoRA") states verbatim: Optimizer Adam; Learning rate
  `1×10⁻⁴` (commonsense) / `2×10⁻⁴` (tool use); Epochs 3; Batch size 16; Maximum
  sequence length 1024 / 4096; **LoRA rank r 16; LoRA scaling α 32; LoRA dropout
  0.05**; Adapted modules "Query, key, value, up-, and down-projections"; Anchor
  corpus "MATH-lighteval (3B); GSM8K (1.5B)"; **Number of anchor examples 512**;
  Protected spectral energy 99%; Reasoning tokens "All tokens in the generated
  reasoning traces". Config defaults and Smoke B now match these; the 3090 batch
  size stays flexible via gradient accumulation (memory constraint, not a paper
  deviation).
* **F3 — null-space share "20–45%" was wrong on both ends.**
  Recomputed from Table 1 values in the paper: QKV (n=2048) per-layer
  1378/765/755/515/407/393 → 67.3/37.4/36.9/25.1/19.9/**19.2%**;
  Down (n=11008) 7028/7343/5474/2272/1701/5940 →
  63.9/66.7/49.7/20.6/**15.5%**/54.0%; Up (n=2048) 1037/1138/574/325/259/251 →
  50.6/55.6/28.0/15.9/12.6/**12.3%**. True per-layer range is **12.3–67.3%**;
  the 29.2–45.1% figures are the three module *averages* (34.3/45.1/29.2%).
  The old text conflated averages with the range and rounded both wrongly.
  Smoke A now expects "same order of magnitude on LFM3", since Table 1 was
  measured on Qwen2.5-3B, not LFM3.
* **F4 — anchor-count wording conflated Fig.3 with Fig.5b.**
  Paper text: Fig.3 plots similarity to the **4096-sample reference** null space
  and "largely stabilizes **beyond 512** samples"; Fig.5b (anchor-size ablation)
  says "Target-task accuracy remains nearly unchanged from **32 to 512** ...
  reasoning retention is stable from **64 to 512** ... degrades mainly at **32**
  examples, particularly for Qwen2.5-3B". The draft wrote "Fig.3/5b stabilizes
  ≥64–512", attributing Fig.5b's interval to Fig.3. Split accordingly; 512 stays
  the default (Table 8) with 64–128 justified for smoke runs by Fig.5b, and 32
  removed from the ablation grid as a known-degraded point.
* **F5 — removed the invented `alpha=r → scale 1` decision.**
  The draft fixed scaling to 1 "to avoid extra knob", but Table 8 explicitly
  lists LoRA scaling α=32 with r=16 (effective factor α/r=2), so scaling to 1
  would silently deviate from the paper. The update is now
  `(alpha/r)·B·C·Nsᵀ` with `r=16, alpha=32` defaults. This also preserves
  Remark 3.1's interpretation (`A = C·Nsᵀ`, row space restricted to the null
  space) — a scalar prefactor does not change the row-space argument.
* **F6 — B/C init relabeled as engineering choice, not paper fact.**
  Full-text search of the paper for initiali*ation returns only JS boilerplate
  and the word "initialization" in related-work prose (O-LoRA/MiLoRA discussion);
  no init is specified for NB-LoRA's B/C. The `B=zeros, C=kaiming` prescription
  is therefore marked as following standard LoRA convention (Hu et al. 2022,
  cited in the paper as PEFT reference [`Hu et al., 2022`]) so that ΔW=0 at step 0;
  here `C·Nsᵀ` plays the role of LoRA's `A` matrix.
* **F7 — scaffold no longer touches `trl/experimental/__init__.py`.**
  Read the file (36 lines): it contains only the license header, docstring, and
  the `TRLExperimentalWarning` block — zero per-trainer re-exports. The import
  pattern is direct submodule import (`from trl.experimental.gmpo import ...`,
  as in `docs/source/paper_index.md:145`), so the nblora package follows suit.
* **F8 — doc filename `nblora_trainer.md` → `nblora.md`.**
  `docs/source/` listing shows the closest analog — GMPO, likewise an
  experimental single-trainer package (`gmpo/{__init__,gmpo_config,gmpo_trainer}.py`)
  — documented as `gmpo.md`, not `gmpo_trainer.md`. Matching that precedent.
* **F9 — `eigh` timing "~seconds" → "tens of seconds to minutes".**
  Symmetric eigendecomposition costs ≈(4/3)n³ flops; at n=7680 that is
  ≈6×10¹¹ flops, i.e. tens of seconds minimum on a single CPU (minutes on
  slower cores), versus genuine seconds at n=2048. Consequence unchanged
  (compute once, cache), but scheduling expectations for the LFM3 `w2` site
  (n=7680) are now realistic.
* **F10 — verified (not changed): Gram-matrix memory math.**
  2048²×4 B = 16,777,216 B = 16.0 MiB; 7680²×4 B = 235,929,600 B ≈ 236 MB fp32 —
  both figures in §3 check out exactly, so CPU-side streaming accumulation
  (`G += H_batchᵀ·H_batch`, valid since `HᵀH = Σ hᵢhᵢᵀ`) stands as designed.
* **F11 — verified (not changed): constraint, merge, sharing claims.**
  Eq.5 `H·ΔWᵀ = H·Ns·Cᵀ·Bᵀ = 0 ∀B,C` and the merged form `W+B·C·Nsᵀ`
  (Algorithm 1, line 12) are quoted verbatim in the paper; §3.2's
  "Q, K, V share the same input hidden states, so the corresponding null space
  can be computed once and reused" supports runtime site-sharing instead of
  name-based sharing. The generic-implementation requirement (§1–§2) is
  compatible: sharing is detected by shared input tensor, never by module name.

## 8. Second from-scratch audit (2026-11-04) — fixes G1–G8

Method: re-read every line of `null.md` against the paper's TeX source
(`arxiv.org/e-print/2609.25618v1`: `content/3_methodology.tex`,
`content/4_experiment.tex`, `content/appendix_2.tex`, `content/appendix_3.tex`)
plus live repo/environment checks. Paper equations quoted from source;
repo facts re-verified by command, not memory.

* **G1 [MAJOR] — the `α/r` scaling factor is NOT in the paper; it is now marked
  as a LoRA-convention engineering choice.** Paper Eq.4
  (`content/3_methodology.tex:126`): `ΔW = B·C·Nsᵀ` — no prefactor.
  Algorithm 1 (`3_methodology.tex:158,160`): `ΔWℓ = Bℓ·Cℓ·Ns,ℓᵀ`,
  `Ensure ... Wℓ + Bℓ·Cℓ·Ns,ℓᵀ` — no prefactor. `grep -rn "alpha\|scaling"`
  over `content/*.tex` finds the ONLY occurrence in the hyperparameter table
  (`appendix_3.tex:22`: `LoRA scaling α = 32`); the methodology never states
  how (or whether) α enters the NB-LoRA update. The prior §7/F5 wording
  ("effective factor α/r=2 ... would silently deviate from the paper") was
  itself an inference: it assumed standard-LoRA `α/r` semantics apply. The
  honest statement is: `alpha=32` is the paper's Table 8 value, but WHETHER to
  multiply by `α/r` is unspecified by the paper. Applying `α/r` follows
  standard LoRA (Hu et al. 2022, the paper's PEFT reference) and is the sane
  default (it decouples the `r` sweep from the update magnitude), but it must
  be labeled as our choice, ablatable (`alpha` sweep), and applied
  consistently in forward AND merge (`W + (α/r)·B·C·Nsᵀ`). Same treatment for
  dropout placement (Table 8 lists `LoRA dropout 0.05`, no placement stated;
  LoRA convention = dropout on adapter input). §2/§3/merge fixed accordingly.
* **G2 — "8-task sweep" → "9-task (8 commonsense + tool-use)".**
  `4_experiment.tex` Table 2 has 9 target-task columns (ARC-C, ARC-E, BoolQ,
  HellaSwag, OpenBookQA, PIQA, SocialIQA, WinoGrande, ToolUse); §4 text adapts
  "to a tool-use task and eight target tasks from the Commonsense Reasoning
  benchmark". The draft's "8-task" dropped the tool-use column. Fixed.
* **G3 — docs step now includes `_toctree.yml` registration.**
  `docs/source/_toctree.yml:116` registers `local: gmpo`; a new `nblora.md`
  without a matching entry would be an orphan page. Step 7 now requires the
  `local: nblora` entry. (Also verified `docs/source/gmpo.md` is the correct
  short-name precedent: 36 lines, `from trl.experimental.gmpo import ...`.)
* **G4 — venv guidance de-duplicated to `~/.venv/bin/python`.**
  The draft named two setups (`~/.venv` AND `uv run` / project `.venv`).
  Re-verified 2026-11-04: `~/.venv/bin/python` = 3.12.13, torch 2.13.0+cu130,
  transformers 5.18.0, peft 0.21.2, datasets 5.1.0, pytest 9.1.1,
  `trl.__file__ = /home/anon/trl/trl/__init__.py`; `/usr/bin/python3` has no
  torch (`import torch` → Traceback). All runs use `~/.venv/bin/python`.
* **G5 — anchor_max_length default clarified (1024, override 4096 for
  tool-use).** `appendix_3.tex:20`: `Maximum sequence length = 1024 | 4096`
  (commonsense | tool-use columns). A single default of 1024 silently
  truncates tool-use prompts (whose JSON API arguments need the longer
  context). Fixed to default-1024 + explicit tool-use override.
* **G6 — ablation grid now names the paper's exact four module choices.**
  `4_experiment.tex` module-selection text + `appendix_2.tex:8` table header:
  {Q,K,V} vs {Q,K,V + MLP-Up} vs {MLP-Up, MLP-Down} vs {Q,K,V,MLP-Up,MLP-Down}.
  The draft's vague "`target_modules` subsets" is now this explicit set, so a
  runner reproduces Fig.5a instead of inventing their own grid. (Note the
  paper never adapts `o_proj` — "all five" = Q,K,V,up,down — consistent with
  the §1 example list excluding `o_proj`.)
* **G7 — `paper_index.md` target pinned to the PEFT section.**
  Verified `docs/source/paper_index.md:1425` = `## Parameter-Efficient
  Fine-Tuning (PEFT)` hosting the LoRA (`2106.09685`) and DoRA entries, and
  `:1295` = `## Supervised Fine-Tuning`. NB-LoRA is a LoRA variant → PEFT
  section is the precise home (draft said "SFT/PEFT section"). Step 6 fixed.
* **G8 — verified (not changed) with fresh evidence, 2026-11-04:**
  (a) LFM3 linears: `grep` on `modeling_lfm3.py` confirms `w1/w3 =
  Linear(2048,7680)`, `w2 = Linear(7680,2048)`, `q_proj = Linear(2048,16*128)`,
  `k/v_proj = Linear(2048,4*128)`, `o_proj`, `gate_proj`, `in/out_proj`,
  `lm_head` + `_tied_weights_keys = {"lm_head.weight":
  "model.embed_tokens.weight"}` — so "w2 input n=7680", "lm_head tied", and
  the §1 module inventory are exact. (b) Input sharing: `q/k/v_proj` all take
  `hidden_states` (`modeling_lfm3.py:245-247`), `w1/w3` both take `x`
  (`:157`) — runtime site-sharing claim holds on LFM3. (c) Config:
  24 layers, hidden 2048, intermediate 7680, 16/4 heads, hybrid
  conv/full/sliding_attention `layer_types`, `tie_word_embeddings=true` —
  all match §1 verbatim. (d) GPU: `nvidia-smi` = RTX 3090 Ti, 23028 MiB —
  "24GB card / <20GB budget" consistent. (e) LFM3 bf16 2-prompt forward peak
  re-measured 3.468GB — the §4 "3.47GB peak" reproduces to 3 decimals.
  (f) `require_*` gates live in `tests/testing_utils.py:48,55`
  (`require_bitsandbytes` at `:48`, `require_peft` at `:55`) and `tests/experimental/` holds
  `test_gmpo_trainer.py` etc. — §3 test conventions stand. (g) `SDFTConfig`
  subclasses `_BaseConfig` (not `SFTConfig`), but that is no counter-precedent:
  SDFT has a custom loop, while NB-LoRA reuses `SFTTrainer`, so
  `NBLoraConfig(SFTConfig)` correctly mirrors `GMPOConfig(GRPOConfig)`
  (`gmpo_config.py:21`). (h) Table 8 anchor corpora
  (`appendix_3.tex:25`: `MATH-lighteval (3B); GSM8K (1.5B)`), lr
  1e-4/2e-4, epochs 3, batch 16, rank 16, energy 99%, 512 anchors,
  `All tokens in the generated reasoning traces` — all §2 values reconfirmed
  against source, not HTML.

## 9. Third from-scratch audit (2026-11-03) — fixes H1–H2, live re-verification

Method: every checkable claim in §§0–6 re-verified from scratch — paper via
fresh `curl` of `https://arxiv.org/html/2609.25618v1` (HTTP 200, 668037 bytes,
91165 chars extracted text; same content as prior audits), repo/model/env via
live commands (not memory). Two fixes; everything else confirmed.

* **H1 [MINOR] — prior wording implied the paper evaluated `o_proj`; it did
  not.** Paper Fig.5a text (verified in fresh extract): "Figure 5(a) compares
  four choices of tunable modules: only the attention Q/K/V projections, only
  the MLP up/down projections ..., Q/K/V plus the MLP up projection, and all
  five projections." Table 8: "Adapted modules: Query, key, value, up-, and
  down-projections." `o_proj` appears in NEITHER the four-way grid NOR the
  adapted set — "all five" = Q,K,V,up,down. The example list
  `["q_proj","k_proj","v_proj","w1","w3","w2"]` already matched; only the
  parenthetical was wrong. Fixed in §1.
* **H2 [MINOR] — `trust_remote_code=True` must go on tokenizer loads too.**
  Live run 2026-11-03 (`AutoModelForCausalLM ... trust_remote_code=True` +
  bare `AutoTokenizer.from_pretrained`) printed the interactive custom-code
  prompt ("Do you wish to run the custom code? [y/N]") plus the
  `lfm3`-type warning — i.e. the MODEL call was fine, the TOKENIZER call
  prompted. §1 now requires the flag on both. (The run still completed:
  logits `(2, 2, 128000)`, peak `3.468009472 GB` — the §4 "3.468GB" figure
  reproduces to 9 significant digits.)
* **H3 — line-by-line re-verification (no change), with fresh evidence:**
  (a) Method core: Eq.4 `ΔW=B·C·Nsᵀ`, Eq.5 `H·ΔWᵀ=H·Ns·Cᵀ·Bᵀ=0 ∀B,C`,
  `mr+rk < mr+rn` ("smaller ... as k<n"), "unconstrained parameters ... any
  standard optimizer (SGD, Adam) without ... projection", Remark 3.1
  (`A=C·Nsᵀ`) — all present verbatim in fresh extract; §0/§2 wording stands.
  (b) Table 8 re-extracted whole: Adam, 1e-4/2e-4, 3 epochs, batch 16,
  1024/4096, r=16, α=32, dropout 0.05, Q,K,V,up,down, MATH-lighteval (3B) /
  GSM8K (1.5B), 512 anchors, 99% energy, "All tokens in the generated
  reasoning traces" — every §2 default reconfirmed; G1's "α listed but
  application unspecified" also reconfirmed (`alpha` occurs ONLY in that
  table). (c) Init silence reconfirmed: `init` hits in fresh extract are JS
  boilerplate, "specialized initialization" (intro), LoRA-Null/MiLoRA
  init-baseline discussion, and Appendix-A math — NO B/C init for NB-LoRA;
  `B=zeros, C=kaiming` stays labeled engineering. (d) Fig.3 vs Fig.5b split
  reconfirmed verbatim ("stabilizes beyond 512 samples" vs "nearly unchanged
  from 32 to 512 ... stable from 64 to 512 ... degrades mainly at 32");
  energy ablation ("robust ... 0.80 to 0.99, but drops sharply at 0.75")
  reconfirmed. (e) Model: config.json re-read (24 layers, 2048/7680,
  Lfm3ForCausalLM + auto_map, bf16, 16/4 heads, hybrid layer_types,
  tie_word_embeddings=true); all 11 linears re-grepped with dims
  (w1/w3: hidden→intermediate, w2: intermediate→hidden, q: 2048→16·128,
  k/v: 2048→4·128, in_proj: hidden→3·hidden); tied lm_head re-grepped
  (`_tied_weights_keys`); q/k/v + w1/w3 input-sharing re-read in forward
  (`hidden_states` / `x`). (f) Repo: `experimental/__init__.py` re-read (36
  lines, warning-only); `gmpo/{__init__,gmpo_config,gmpo_trainer}.py`
  re-listed; `GMPOConfig(GRPOConfig)` + `SFTConfig`/`SFTTrainer` existence
  re-grepped; `paper_index.md` PEFT (`:1425`, LoRA/DoRA) + SFT (`:1295`)
  re-grepped; `gmpo.md` 36 lines + `_toctree.yml:116` re-checked;
  `tests/experimental/` (incl. `test_gmpo_trainer.py`) +
  `testing_utils.py:48,55` (`require_bitsandbytes`, `require_peft`) +
  `tiny-Qwen2ForCausalLM-2.5` (`test_sft_trainer.py:301`) +
  `AGENTS.md:16,63,68,120` (paper_index rule, hasattr ban, mock ban, HF-link
  rule) re-grepped. (g) Env: `~/.venv` = Python 3.12.13, torch 2.13.0+cu130,
  transformers 5.18.0, peft 0.21.2, datasets 5.1.0, pytest 9.1.1,
  trl→repo — all re-printed live; `/usr/bin/python3 -c "import torch"` →
  Traceback reconfirmed; `nvidia-smi` = RTX 3090 Ti, 23028 MiB reconfirmed.
  (h) Gram math recomputed: 2048²·4 = 16,777,216 B; 7680²·4 = 235,929,600 B;
  eigh cost (4/3)·7680³ ≈ 6.0e11 flops — §3/§6 figures exact.
  No other claim in §§0–6 was found wrong.

## 10. Fourth from-scratch audit (2026-11-04) — fixes I1–I5, repro-verified

Method: re-read every line of `null.md` against the paper TeX source
(`nb_src/content/*.tex`), the landed implementation
(`trl/experimental/nblora/{__init__,nblora_config,nblora_trainer}.py`,
`docs/source/nblora.md`, `docs/source/paper_index.md:1457`,
`docs/source/_toctree.yml:124`, `tests/experimental/test_nblora.py`), and live
env/model commands. Paper claims (§§7–9) all reconfirmed — no paper fix needed.
Found one implementation bug (repro-verified before/after) plus five
spec-vs-implementation drifts in `null.md`; fixed both.

* **I1 [BUG, repro-verified] — `compute_constraint_residual` measured only the
  last probe prompt.** The method looped `for text in probe_texts:
  model(...)` overwriting `captured[name]` each iteration, then computed the
  worst-case ratio once after the loop. Repro (tiny-Qwen2, `B ~ N(0,1)` so the
  residual is prompt-dependent): `r(first)=4.41449`, `r(second)=4.39934`,
  `r(first-two)=4.39934` (= second only, first ignored). After the fix
  (`trl/experimental/nblora/nblora_trainer.py`, per-prompt worst inside the
  loop): `r(first-two)=4.41449 = max(r1, r2)`. Docstring updated to "over
  wrapped modules and probe prompts". Zero-init behavior unchanged (`0.0`);
  all 48 tests in `tests/experimental/test_nblora.py` pass after the fix.
* **I2 — `nblora_anchor_dataset` type omitted `list[str]`.** §2 said
  "dataset (or `str` path)"; the implementation (`_resolve_anchor_texts`)
  accepts `Dataset | list[str] | str` (list covered by
  `test_resolve_anchor_texts_from_list`). §2 fixed to
  "`Dataset` or `list[str]` or `str` path/id".
* **I3 — cache format oversimplified.** §2 said "save/load `{site: Ns}`";
  the implementation saves/loads `{energy_threshold, site_of, null_bases}`
  with validation against current targets/energy/input dims (stale cache
  rebuilds with a warning). §2 fixed.
* **I4 — helper count understated.** §2 said "two small functions ...
  ~60 lines total"; the trainer file has `merge_nblora_weight`,
  `NBLoraLinear`, `_collect_gram_matrices`, `_build_null_bases`,
  `_resolve_anchor_texts` (the two Gram/null-basis functions alone are
  ~100 lines). §2 fixed. `grep -rn "hasattr|getattr"` over
  `trl/experimental/nblora/` returns nothing — the no-`hasattr` rule holds.
* **I5 — `lm_head` wording.** §1 said "Exclude `lm_head` ... only if the user
  lists it — the code must not hardcode any exclusion list." The
  implementation rejects `lm_head` with an error (fused LM head reads
  `lm_head.weight` directly, so the adapter would get no gradients;
  `test_lm_head_target_rejected`), which is a functional guard rather than
  a silent model-specific exclusion. §§1–2 fixed to document the error.
* **I6 [MINOR] — §8/G8f line range.** Said `testing_utils.py:48-52` hosts
  `require_peft`; `require_peft` is at `:55` (`require_bitsandbytes` at
  `:48`). Fixed to `:48,55`, matching §9/H3. Live `grep` reconfirmed.

## 11. Fifth from-scratch audit (2026-11-04) — fix J1, repro-verified

Method: re-read `null.md` against the TeX source (`nb_src/content/*.tex`),
the landed implementation, and live env/repo/model commands; ran repro scripts
for Gram streaming, null-basis orthonormality/tail energy, `NBLoraLinear`
forward/merge/zero-init, Q/K/V site sharing, merge-vs-adapter logits,
`lm_head` rejection, flat-spectrum skip, and cache roundtrip before fixing.

* **J1 [MINOR, repro-verified] — `NBLoraConfig` cache-format docstring was
  stale.** Both the class docstring and the field `help` said the cache file
  holds "`{site: Ns}`". Live repro (tiny-Qwen2, cache roundtrip) shows the
  file holds `{energy_threshold, site_of, null_bases}` (repro printed
  `cache keys: ['energy_threshold', 'null_bases', 'site_of']`). Fixed both
  strings in `trl/experimental/nblora/nblora_config.py` to
  "`{energy_threshold, site_of, null_bases}`". `null.md` §2 already used the
  corrected wording — no plan change needed.
* **Re-verified (no change), with fresh repro evidence:** Gram streaming
  `G += H^T H` equals direct `H_cat^T H_cat` exactly; `_build_null_bases` tail
  keeps <1% energy at `eps=0.99` on a decaying spectrum and skips cleanly on
  flat/empty Grams (`k=0` warning path, matching
  `test_default_energy_gracefully_skips_flat_spectrum`); `NBLoraLinear`
  forward matches manual `xW^T + (α/r)((xNs)C^T)B^T`, `ΔW=0` at init
  (`B=zeros`), `merge_nblora_weight` matches `merged_weight()`; Q/K/V share
  one site per layer (2 sites for 6 targets on tiny-Qwen2), merge reproduces
  adapter logits and leaves zero `nblora_` keys; `lm_head` targeting raises
  `ValueError`; paper TeX re-grepped (`ΔW=B·C·Ns^T` no prefactor,
  Q/K/V-sharing quote, zero `o_proj` hits, anchor/energy wording, Table 8
  values); repo/env re-verified (`__init__.py` 36 lines, `GMPOConfig(GRPOConfig)`,
  `_toctree.yml:116/124`, `paper_index.md:1295/1425/1457`, `testing_utils.py:48,55`,
  `~/.venv` 3.12.13/torch 2.13.0+cu130/transformers 5.18.0/peft 0.21.2/datasets 5.1.0/pytest 9.1.1,
  3090 Ti 23028 MiB, Gram bytes 16777216/235929600). All 48 tests pass after J1.
