# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-09-30

### Fixed
- `health()`, `info()` and `llmreport health` crashed on dynamically quantized INT8 models (such as an exported
  `model_int8.pt`). Quantized weights are now checked like any others.
- Quantized left-to-right models (such as an INT8 GPT) were labelled "encoder-only (bidirectional)". The
  left-to-right test now runs on temporary float copies of quantized layers, whose activation scaling
  otherwise adds noise to every position.
- The typo-robustness summary no longer says typos "cancel -5%" of the context benefit; it says typos
  don't reduce it.

### Added
- `python -m llmreport ...` runs the command-line tool, for when the `llmreport` command is not on PATH
  (common with `pip install --user` on Windows).

## [0.3.0] - 2026-09-30

### Added
- Quick functions in the style of `np.sum`: `params`, `param_breakdown`, `architecture`, `memory`, `info`,
  `health`, `grad_norm`, `weight_norm`, `perplexity`, `generate`, `predict_next`, `speed`, `code_score`,
  `text_stats`, `check` and `compare`.
- `llmreport.info()`: everything about a model at this moment, including while it trains (mode, frozen
  parameters, gradient and weight norms, optimizer, learning rate and optimizer memory).
- `llmreport.health()`: finds NaN/inf weights and gradients, all-zero layers, collapsed norms, dead neurons,
  extreme and outlier layers, float16 overflow risk, and parameters that get no gradient.
- `llmreport.Tracker` for live training: loss, smoothed loss, learning rate, gradient norm, speed and GPU memory
  every step; periodic perplexity, sample generations and health checks; alerts for NaN, spikes, exploding or
  vanishing gradients, plateaus, overfitting and slowdowns; `metrics.jsonl`/`events.jsonl` logs; a
  self-refreshing HTML dashboard; a live Jupyter panel; resuming; a Hugging Face `Trainer` callback.
- `llmreport.track_checkpoints()` charts progress across saved checkpoints; `llmreport.load_run()` reads runs.
- CLI commands: `watch`, `dashboard`, `info`, `health` and `progress`.
- More kinds of model: encoder-decoder (T5, BART), masked (BERT, RoBERTa, with pseudo-perplexity) and base
  models without a language-model head. Checks that don't apply are marked "skipped" with the reason.
- Whether a model is left-to-right or bidirectional is measured directly; encoder and decoder layers are counted
  separately.

### Fixed
- RoBERTa-style models: the context window accounts for the two reserved position slots.

## [0.2.0] - 2026-09-29

### Added
- Works with any PyTorch language model, not just Hugging Face ones: from-scratch GPTs, Llama-style models,
  recurrent and state-space models, TorchScript modules, INT8 dynamically quantized models, or a plain
  function from token ids to logits. llmreport finds out how to call the model and where the logits are.
- Built-in greedy generation for models without a Hugging Face-style `generate()`, stopping at end-of-text tokens.
- Any tokenizer: Hugging Face, `tokenizers.Tokenizer` or a `tokenizer.json` path, SentencePiece, tiktoken, or any
  object with `encode`/`decode`.
- `llmreport.load_checkpoint()` for plain PyTorch checkpoints (state dicts with stored configs, whole models,
  `torch.compile`/DataParallel prefixes, TorchScript, safetensors, quantized weights).
- `prompt_template` for instruction-tuned models without a chat template, plus `forward`, `context_length`,
  `generation`, `model_class` and `stop_tokens` options. The CLI gains `--model-class`, `--tokenizer`,
  `--template`, `--context-length` and `--stop-tokens`.
- Architecture check reads the structure from the modules: family, attention type (MHA/GQA/MQA) and KV heads,
  feed-forward type (standard, gated, mixture of experts), position encoding, norm type and pre/post placement,
  activation, weight tying, parameters without position embeddings, and a warning when the tokenizer doesn't
  match the model.
- New `code` check: writes small Python functions and runs them against unit tests (pass@1).
- `LanguageModel` and `TokenizerAdapter` are public, so custom checks work with every kind of model.

### Changed
- Perplexity for non-Hugging Face models is computed from the logits; for Hugging Face models it still uses the
  model's own loss. Both agree with a hand calculation.

## [0.1.0] - 2026-09-29

### Added
- `llmreport.analyze()` one-call entry point for Hugging Face causal language models.
- Analyzers: architecture, performance, perplexity, repetition, consistency, robustness, refusal.
- `Report` object with notebook rendering, terminal output, and HTML, Markdown (model card) and JSON export.
- `Report.compare()` for before/after comparisons (for example, before and after fine-tuning).
- Plugin system: `register_analyzer` for custom checks.
- `llmreport` command-line tool.
- `analyze()` accepts a model name or folder, model and tokenizer, a model alone (tokenizer found automatically), a `(model, tokenizer)` tuple, or a text-generation pipeline; checks can be given as a string (`"a,b"`).
- Consistency and typo robustness score correctness on factual questions with known answers, and say "can't be judged" when outputs are loops or never correct.
- Known-answer test suite (published parameter counts, KV-cache sizes, uniform-model perplexity, timed model, scripted behavior models).
