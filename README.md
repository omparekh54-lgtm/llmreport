# llmreport

**A one-line behavior and performance report for any language model.**

You trained or fine-tuned an LLM. Now what is it actually like? `llmreport` runs a set of quick checks and gives you one readable report, right in your notebook:

```python
import llmreport

report = llmreport.analyze(model, tokenizer)   # or: llmreport.analyze("gpt2")
report                                          # renders as a report in Jupyter / Colab
```

It works with **any PyTorch language model**: Hugging Face models, and models you wrote yourself (nanoGPT-style GPTs, Llama-style models with RoPE and grouped-query attention, recurrent LSTMs, state-space models, TorchScript files, INT8-quantized checkpoints). The only requirement is that calling the model on token ids returns next-token scores.

| Check | What it tells you |
|---|---|
| **Architecture** | Read from the model's actual modules: family (transformer, recurrent, state-space, mixture of experts), layers, hidden size, attention heads and KV heads (MHA / GQA / MQA), feed-forward size and type (standard, gated/SwiGLU, experts), position encoding (learned, RoPE, ALiBi), norm type and pre/post placement, activation, weight tying, parameters and where they are, memory at FP32/FP16/INT8/INT4, KV-cache size, tokenizer facts |
| **Performance** | Time to first token and tokens/second at several prompt lengths, GPU and process memory |
| **Perplexity** | How well it predicts everyday English, plus tokenizer-independent bits per character |
| **Repetition** | Whether it gets stuck in loops (repeated 3-grams, distinct-1/2) |
| **Consistency** | Whether simple factual questions get the same (correct) answer when reworded |
| **Typo robustness** | How much of the benefit of context typos destroy, and whether misspelled questions are still answered correctly |
| **Refusals** | Whether it declines harmful requests, and whether it over-refuses harmless-but-edgy ones |
| **Python code** | Writes small Python functions and runs them against unit tests (pass@1), plus how often the output is valid Python |

Every check gives a status (OK / Check / Info), a plain-English summary, the numbers, notes on caveats, and the actual sample outputs so you can judge for yourself.

## Install

```bash
pip install llmreport
```

Or straight from GitHub (latest code, before a PyPI release):

```bash
pip install git+https://github.com/omparekh54-lgtm/llmreport.git
```

In Jupyter or Colab, put `!` in front: `!pip install llmreport`.

Requires Python 3.9+, PyTorch and `transformers`. Add `pip install "llmreport[rich]"` for nicer terminal tables. SentencePiece and tiktoken tokenizers work if those packages are installed.

## Usage

### Every way to call it

`analyze` accepts whatever you already have:

```python
import llmreport

llmreport.analyze("gpt2")                       # a model name on the Hugging Face Hub
llmreport.analyze("./my-finetuned-model")       # a local folder
llmreport.analyze(model, tokenizer)             # model and tokenizer objects
llmreport.analyze(model)                        # tokenizer found automatically from the model's name/folder
llmreport.analyze(pipe)                         # a transformers text-generation pipeline
llmreport.analyze(llmreport.load("gpt2"))       # a (model, tokenizer) pair
llmreport.analyze(my_model, "tokenizer.json")   # your own nn.Module + a tokenizer file
llmreport.analyze("final.pt", "tokenizer.json", model_class=GPT)   # a checkpoint file

from llmreport import analyze                   # or import just the function
report = analyze("gpt2", checks="perplexity")   # one check, a "a,b" string, or a list
```

### Your own model (written from scratch)

Pass your model and its tokenizer. llmreport works out how to call the model and where the logits are, reads the architecture from the modules, and generates text with its own greedy loop if the model has no Hugging Face-style `generate()`:

```python
from gpt_model import GPT            # your model class
import llmreport

model, tokenizer = llmreport.load_checkpoint(
    "checkpoints/final.pt", GPT,     # {"model": state_dict, "config": cfg} or a whole saved model
    tokenizer="tokenizer.json",
)
report = llmreport.analyze(
    model, tokenizer,
    prompt_template="<|instruction|>{prompt}<|response|>",   # how your instruction-tuned model expects prompts
)
report.to_html("my_model.html")
```

What it handles automatically:

- **Calling the model**: `model(ids)`, `model(input_ids=ids)`, outputs that are a tensor, a `(logits, loss)` tuple, a dict, or an object with `.logits`; `(seq, batch, vocab)` layouts; nanoGPT-style models that only return the last position unless given targets.
- **Tokenizers**: Hugging Face tokenizers, `tokenizers.Tokenizer` objects or `tokenizer.json` files, SentencePiece (`.model`), tiktoken, or any object with `encode` and `decode`. End-of-text tokens such as `<|endoftext|>`, `</s>` or `<|eot_id|>` are found and used to stop generation.
- **Checkpoints** (`load_checkpoint`): whole saved models, state dicts with or without a stored config (dataclass, dict, argparse namespace), `torch.compile` / DataParallel prefixes, Lightning-style `state_dict` keys, TorchScript files, `.safetensors`, and dynamically quantized INT8 weights. Only load checkpoints you trust: files that store Python objects are read with pickle.
- **Context window**: from the config (`block_size`, `max_seq_len`, `max_position_embeddings`, ...) or the size of the position embedding. Set `context_length=` if it can't be found.

If your model needs a special call, give llmreport a function from token ids to logits:

```python
llmreport.analyze(model, tokenizer, forward=lambda ids: model(ids, start_pos=0))
```

### In a notebook

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import llmreport

model = AutoModelForCausalLM.from_pretrained("my-org/my-model")
tokenizer = AutoTokenizer.from_pretrained("my-org/my-model")

report = llmreport.analyze(model, tokenizer)
report
```

### Choose checks and settings

```python
llmreport.list_checks()                              # see what is available

report = llmreport.analyze(
    model, tokenizer,
    checks=["architecture", "perplexity", "repetition"],
    skip="performance",                              # leave checks out
    mode="full",                                     # "quick" (default) or "full"
    options={"performance": {"prompt_lengths": [128, 1024], "repeats": 5}},
)
```

Typos in check names get a suggestion (`Unknown check ['perplexity_check'] ... Did you mean ['perplexity']?`), and passing a model that can't generate text gives a clear error instead of a stack trace deep inside PyTorch.

### Use your own prompts

```python
report = llmreport.analyze(
    model, tokenizer,
    prompts={
        "perplexity":  ["Text from your domain...", "More text..."],
        "repetition":  ["Write a product description for ..."],
        "consistency": [["What is our refund policy?", "How do refunds work here?"]],
        "refusal":     {"should_refuse": [...], "should_answer": [...]},
    },
)
```

### Export and share

```python
report.to_html("report.html")          # standalone page, works in light and dark mode
report.to_markdown("MODEL_CARD.md")    # Hugging Face-style model card with TODOs for you to fill in
report.to_json("report.json")          # raw numbers and samples

report.metrics                         # {"perplexity.perplexity": 21.3, ...}
report["repetition"].details["samples"]
```

### Compare before and after fine-tuning

```python
before = llmreport.analyze(base_model, tokenizer)
after  = llmreport.analyze(finetuned_model, tokenizer)
before.compare(after, names=("base", "fine-tuned"))
```

Each metric is marked **better** or **worse** where the direction is known.

### Command line

```bash
llmreport gpt2 --html gpt2.html --markdown MODEL_CARD.md
llmreport my-org/my-model --checks architecture,performance --device cuda
llmreport checkpoints/final.pt --model-class gpt_model:GPT --tokenizer tokenizer.json \
          --template "<|instruction|>{prompt}<|response|>" --html report.html
llmreport --list-checks
```

`--model-class` takes `module:Class` (run it from the folder that contains the module) or `path/to/file.py:Class`.

## Write your own check

```python
import llmreport

@llmreport.register_analyzer
class AnswerLength(llmreport.Analyzer):
    name = "answer_length"
    title = "Answer length"
    description = "Average length of answers in words."

    def run(self, model, tokenizer, config):
        # model and tokenizer work the same for every kind of model:
        # model.logits(ids), model.generate_ids(ids, n), tokenizer.encode(text), tokenizer.decode(ids)
        from llmreport._utils import generate
        prompts = ["What is AI?", "Describe a cat."]
        lengths = [len(generate(model, tokenizer, p, 64, config).split()) for p in prompts]
        avg = sum(lengths) / len(lengths)
        return self.result(f"Answers average {avg:.0f} words.", metrics={"avg_words": avg})

report = llmreport.analyze(model, tokenizer, checks=["architecture", "answer_length"])
```

If one check crashes, the others still run and the error shows up in the report.

## How to read the results (honest limits)

- These are **quick indicators on small prompt sets**, not a full evaluation. Use them to spot problems fast, then dig deeper with tools such as `lm-evaluation-harness` (benchmarks) or `garak` / PyRIT (red-teaming).
- A check says **"can't be judged"** instead of guessing when the evidence is missing: for example, consistency is not scored if the model's answers are loops, or if it answers none of the factual questions correctly.
- Answers count as correct when they *mention* an accepted answer ("Paris"), so a rambling answer that includes it still counts.
- Refusal detection matches phrases like "I can't help" or "I'm sorry, but" near the start of the answer. Base models (no chat template) usually refuse nothing, and the report says so.
- Perplexity depends on the tokenizer. Compare bits per character across models with different tokenizers.
- Behavior checks use greedy decoding, which repeats more than typical sampling settings.
- For models without a Hugging Face `generate()`, llmreport generates with its own loop that re-runs the whole sequence for each new token (no KV cache). Results are exact; speeds are what a model without a cache achieves.
- The **Python code** check runs the model's code in a separate Python process with a time limit. Turn that off with `options={"code": {"execute": False}}` to only check syntax. Instruction-tuned models (chat template or `prompt_template`) get a request; base models continue a function signature and docstring.
- Checks written for English (perplexity, consistency, typo robustness, refusals) will score a code-only model badly. That reflects what it was trained on, not a bug.
- `compare()` ignores tiny differences (under 1 percentage point, or under 10% for timings) so run-to-run noise isn't labelled better or worse.
- Results are reproducible with the same `seed`, model, and hardware.

## How the analysis is tested

The test suite checks the numbers against cases where the right answer is known in advance:

- **Parameter counts** match the published sizes of GPT-2 small (124,439,808), Llama-2-7B (6,738,415,616) and Mistral-7B (7,241,732,096), with the attention / MLP / norm / embedding split checked by hand. These models are built with empty weights, so the test downloads nothing.
- **KV-cache size** matches the known 512 KiB per token for Llama-2-7B in FP16, and 128 KiB for Mistral-7B with grouped-query attention.
- **Perplexity** equals the vocabulary size exactly for a model that treats every token as equally likely, and matches an independent hand calculation.
- **Speed**: a model slowed to 20 ms per step is measured at about 50 tokens per second.
- **Models written from scratch**: a nanoGPT-style GPT (the PyCoder model this library was first built for) reports exactly its published 142,627,840 parameters; Llama-style (RoPE, RMSNorm, grouped-query attention, SwiGLU), LSTM, sequence-first `nn.Transformer`, TorchScript and INT8-quantized models are each checked for the right structure and for perplexity equal to a hand calculation from their logits.
- **Same weights, two implementations**: a from-scratch GPT and the same weights loaded into Hugging Face GPT-2 give the same perplexity, the same architecture numbers and token-for-token the same generated text.
- **Hugging Face architectures** (Llama, Qwen2, GPT-NeoX, Gemma, Mixtral, Mamba) get the right attention type, feed-forward type and activation.
- **Tokenizers**: Hugging Face, `tokenizers`, tokenizer.json, SentencePiece, tiktoken and a hand-written character tokenizer all give perplexity equal to the vocabulary size on a uniform model.
- **Behavior checks** are run on scripted models whose answers we write ourselves (always correct, correct in only one phrasing, looping, refusing everything, never refusing, typo-tolerant, typo-brittle), and each must reach the expected verdict.

## Development

```bash
git clone https://github.com/omparekh54-lgtm/llmreport
cd llmreport
pip install -e ".[dev,rich]"
pytest            # runs offline in seconds with a tiny random model
ruff check src tests
```

## Roadmap

- Calibration check (does confidence match accuracy?)
- Long-context "needle in a haystack" check
- Optional wrappers for `lm-evaluation-harness` and `garak`
- Charts for speed vs prompt length
- Support for API-based models
- KV-cache-aware generation for custom models

Contributions are welcome, and new checks are a great first PR.

## License

MIT
