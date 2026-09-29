"""Can the model write small, working Python functions?

Each task has unit tests. The model's code is checked for valid syntax and then run
against the tests in a separate Python process with a time limit, so a pass means the
code actually works (pass@1 with greedy decoding).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

from .._utils import build_prompt, load_data, mean, uses_instruction_format
from ..adapters import as_model, as_tokenizer
from .base import INFO, OK, WARNING, Analyzer, RunConfig

_FENCE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def completion_prompt(problem) -> str:
    return f'{problem["signature"]}\n    """{problem["doc"]}"""\n'


def truncate_body(completion: str) -> str:
    """Keep the indented function body; stop at the first line that starts a new top-level statement."""
    kept = []
    for line in completion.split("\n"):
        if line.strip() and not line.startswith((" ", "\t")):
            break
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def extract_function(text: str, name: str) -> str:
    """Pull the function ``name`` (plus any imports) out of a free-form answer."""
    fenced = _FENCE.findall(text)
    if fenced:
        text = next((block for block in fenced if f"def {name}" in block), fenced[0])
    lines = text.replace("\r\n", "\n").split("\n")
    start = next((i for i, line in enumerate(lines) if re.match(rf"\s*def\s+{re.escape(name)}\s*\(", line)), None)
    if start is None:
        start = next((i for i, line in enumerate(lines) if re.match(r"\s*def\s+\w+\s*\(", line)), None)
    if start is None:
        return text.strip() + "\n"
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = [lines[start][indent:]]
    for line in lines[start + 1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line[indent:] if line[:indent].strip() == "" else line)
    imports = [line.strip() for line in lines[:start] if re.match(r"\s*(import|from)\s+\w", line)]
    return "\n".join(imports + body).rstrip() + "\n"


def check_syntax(code: str, name: str) -> Tuple[bool, bool]:
    """(valid Python?, defines a function called ``name``?)"""
    if not code.strip():
        return False, False  # an empty answer is not a program
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False, False
    defined = any(isinstance(node, ast.FunctionDef) and node.name == name for node in ast.walk(tree))
    return True, defined


def _limits():  # pragma: no cover - runs in the child process
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    except Exception:
        pass


def run_tests(code: str, tests: List[str], timeout: float = 10.0) -> Tuple[bool, str]:
    """Run ``code`` followed by the ``tests`` in a fresh, isolated Python process."""
    program = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path], cwd=tmp, capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limits if os.name == "posix" else None, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return False, "timed out"
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr or "").strip().splitlines()
    return False, err[-1] if err else f"exit code {proc.returncode}"


class CodeAnalyzer(Analyzer):
    name = "code"
    title = "Python code"
    description = "Writes small Python functions and runs them against unit tests (pass@1)."
    needs_generation = True

    def run(self, model, tokenizer, config: RunConfig):
        tok = as_tokenizer(tokenizer)
        lm = as_model(model, tok)
        problems = config.prompts.get(self.name) or load_data("code")["problems"]
        problems = config.sample(problems, 6)
        max_new = config.gen_tokens(128, 256)
        style = config.option(self.name, "style", "auto")
        if style == "auto":
            style = "instruction" if uses_instruction_format(tok, config) else "completion"
        if style not in ("instruction", "completion"):
            raise ValueError("code style must be 'auto', 'instruction' or 'completion'")
        execute = config.option(self.name, "execute", True)
        timeout = config.option(self.name, "timeout", 10.0)

        rows = []
        for p in problems:
            if style == "completion":
                prompt = completion_prompt(p)
                ids = tok.encode(prompt, add_special=True)
            else:
                text, add_special = build_prompt(tok, p["instruction"], config)
                ids = tok.encode(text, add_special=add_special)
            output = tok.decode(lm.generate_ids(ids, max_new), skip_special=True)
            if style == "completion":
                body = truncate_body(output)
                code = prompt + body
                # The signature and docstring alone are valid Python, so an empty body doesn't count.
                valid, defined = check_syntax(code, p["name"]) if body.strip() else (False, False)
            else:
                code = extract_function(output, p["name"])
                valid, defined = check_syntax(code, p["name"])
            passed: Optional[bool] = None
            error = ""
            if execute and valid and defined:
                passed, error = run_tests(code, p["tests"], timeout)
            elif execute:
                passed = False
                error = ("empty function body" if style == "completion" and not code[len(prompt):].strip()
                         else "not valid Python" if not valid else f"no function named {p['name']}")
            rows.append({"task": p["name"], "output": output, "code": code, "valid_syntax": valid,
                         "defines_function": defined, "passed": passed, "error": error})

        n = len(rows)
        syntax = mean(r["valid_syntax"] for r in rows)
        defined = mean(r["defines_function"] for r in rows)
        metrics = {"syntax_valid_rate": round(syntax, 4), "defines_function_rate": round(defined, 4)}
        notes = [
            f"{'Instruction' if style == 'instruction' else 'Completion'} style: "
            + ("the model gets a request such as \"Write a Python function add(a, b) ...\"."
               if style == "instruction" else "the model continues a function signature and docstring "
               "(HumanEval style)."),
            "Greedy decoding, one attempt per task (pass@1). These are small tasks: a good score shows basic "
            "competence, not that the model is ready for real projects.",
        ]
        if execute:
            pass_rate = mean(bool(r["passed"]) for r in rows)
            metrics["pass_rate"] = round(pass_rate, 4)
            passed_n = sum(bool(r["passed"]) for r in rows)
            summary = (f"Wrote working code for {passed_n} of {n} small Python tasks ({pass_rate:.0%} pass@1); "
                       f"{syntax:.0%} of answers were valid Python.")
            if pass_rate >= 0.5:
                status = OK
            elif pass_rate >= 0.2:
                status = INFO
            else:
                status = WARNING
                notes.append("A low score is expected for models that weren't trained on code.")
            notes.append("The generated code ran in a separate Python process with a time limit. Turn this off "
                         "with options={\"code\": {\"execute\": False}}.")
        else:
            status = OK if syntax >= 0.5 else INFO
            summary = (f"{syntax:.0%} of answers to {n} small Python tasks were valid Python and "
                       f"{defined:.0%} defined the requested function (tests not run).")
        metrics["tasks"] = n
        return self.result(summary, metrics=metrics, details={"style": style, "samples": rows},
                           status=status, notes=notes)
