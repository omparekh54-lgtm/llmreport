# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

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
