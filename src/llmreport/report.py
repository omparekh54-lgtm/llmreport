"""The Report object: notebook display, terminal text and HTML / Markdown / JSON export."""

from __future__ import annotations

import html
import json
import math
from typing import Any, Dict, Iterator, List, Optional

from ._utils import human_bytes, human_number
from .analyzers.base import AnalyzerResult

STATUS_LABEL = {"ok": "OK", "warning": "Check", "info": "Info", "skipped": "Skipped", "error": "Error"}
STATUS_ICON = {"ok": "✓", "warning": "!", "info": "i", "skipped": "–", "error": "✕"}

# Direction used by Report.compare(): +1 means higher is better, -1 lower is better.
METRIC_DIRECTION = {
    "perplexity": -1,
    "bits_per_char": -1,
    "repeated_3gram_rate": -1,
    "looping_outputs": -1,
    "ttft_ms": -1,
    "ttft_ms_longest_prompt": -1,
    "typo_perplexity_increase": -1,
    "over_refusal_rate": -1,
    "decode_tokens_per_s": 1,
    "decode_tokens_per_s_longest_prompt": 1,
    "distinct_1": 1,
    "distinct_2": 1,
    "consistency_score": 1,
    "accuracy": 1,
    "correct_answer_agreement": 1,
    "degenerate_answers": -1,
    "typo_context_benefit_lost": -1,
    "typo_correct_retained": 1,
    "clean_accuracy": 1,
    "typo_accuracy": 1,
    "typo_answer_stability": 1,
    "harmful_refusal_rate": 1,
    "pass_rate": 1,
    "syntax_valid_rate": 1,
    "defines_function_rate": 1,
}


# ----------------------------------------------------------------- formatting


def format_value(key: str, value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, (int, float)):
        if key.endswith("_bytes") or "_bytes_" in key:
            return human_bytes(value)
        if key.endswith("params") or key.startswith("params_") or key in ("tokens_scored",):
            return human_number(value)
        if key.endswith("_rate") or key in (
            "accuracy", "correct_answer_agreement", "answer_overlap", "unrelated_baseline",
            "typo_perplexity_increase", "typo_context_benefit_lost", "typo_correct_retained", "context_benefit",
            "typo_answer_overlap", "clean_accuracy", "typo_accuracy", "distinct_1", "distinct_2",
        ):
            return f"{value:.1%}"
        if isinstance(value, float):
            return f"{value:,.4g}" if abs(value) < 1000 else f"{value:,.1f}"
        return f"{value:,}"
    return str(value)


_KEY_WORDS = {
    "ttft": "time to first token", "est": "estimated", "pct": "%", "kv": "KV", "3gram": "3-gram",
    "rss": "RSS", "gpu": "GPU", "fp32": "FP32", "fp16": "FP16", "int8": "INT8", "int4": "INT4",
    "ms": "(ms)", "s": "(s)", "nll": "NLL",
}


def pretty_key(key: str) -> str:
    parts = [_KEY_WORDS.get(p, p) for p in key.split("_")]
    text = " ".join(parts).replace("per (s)", "per second")
    return text[:1].upper() + text[1:]


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _truncate(text: str, n: int = 240) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


# ----------------------------------------------------------------- HTML pieces

_CSS = """
.llmr{--llmr-fg:var(--jp-content-font-color1,#1f2328);--llmr-muted:var(--jp-content-font-color2,#59636e);
--llmr-bg:var(--jp-layout-color0,#ffffff);--llmr-card:var(--jp-layout-color1,#f6f8fa);
--llmr-border:var(--jp-border-color2,#d1d9e0);--llmr-ok:#1a7f37;--llmr-warn:#9a6700;--llmr-err:#cf222e;
--llmr-info:#0969da;--llmr-bar:#8250df;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--llmr-fg);font-size:14px;line-height:1.5;max-width:980px}
@media (prefers-color-scheme:dark){.llmr.llmr-standalone{--llmr-fg:#e6edf3;--llmr-muted:#9198a1;--llmr-bg:#0d1117;
--llmr-card:#151b23;--llmr-border:#3d444d;--llmr-ok:#3fb950;--llmr-warn:#d29922;--llmr-err:#f85149;--llmr-info:#4493f8;--llmr-bar:#ab7df8}}
.llmr *{box-sizing:border-box}
.llmr h2{font-size:20px;margin:0 0 4px}
.llmr .llmr-meta{color:var(--llmr-muted);font-size:12.5px;margin-bottom:14px}
.llmr .llmr-score{border:1px solid var(--llmr-border);border-radius:8px;overflow:hidden;margin-bottom:18px}
.llmr .llmr-row{display:grid;grid-template-columns:150px 78px 1fr;gap:10px;padding:9px 12px;
border-top:1px solid var(--llmr-border);align-items:start}
.llmr .llmr-row:first-child{border-top:none}
.llmr .llmr-name{font-weight:600}
.llmr .llmr-badge{display:inline-block;font-size:11.5px;font-weight:600;padding:1px 8px;border-radius:10px;
border:1px solid currentColor;white-space:nowrap}
.llmr .s-ok{color:var(--llmr-ok)}.llmr .s-warning{color:var(--llmr-warn)}.llmr .s-error{color:var(--llmr-err)}
.llmr .s-info,.llmr .s-skipped{color:var(--llmr-info)}
.llmr details{border:1px solid var(--llmr-border);border-radius:8px;margin:10px 0;background:var(--llmr-bg)}
.llmr summary{cursor:pointer;padding:10px 12px;font-weight:600;list-style:none}
.llmr summary::-webkit-details-marker{display:none}
.llmr summary::before{content:"▸ ";color:var(--llmr-muted)}
.llmr details[open]>summary::before{content:"▾ "}
.llmr .llmr-body{padding:0 14px 14px}
.llmr .llmr-sum{margin:0 0 10px}
.llmr table{border-collapse:collapse;width:100%;margin:8px 0 12px;font-size:13px}
.llmr th,.llmr td{border:1px solid var(--llmr-border);padding:5px 8px;text-align:left;vertical-align:top;
overflow-wrap:anywhere;min-width:70px}
.llmr th{background:var(--llmr-card);font-weight:600;overflow-wrap:normal}
.llmr td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.llmr .llmr-notes{color:var(--llmr-muted);font-size:12.5px;margin:6px 0 0;padding-left:18px}
.llmr h4{font-size:13px;margin:12px 0 4px;color:var(--llmr-muted);text-transform:uppercase;letter-spacing:.04em}
.llmr .llmr-bar{height:10px;background:var(--llmr-bar);border-radius:3px;min-width:2px}
.llmr pre{background:var(--llmr-card);padding:8px;border-radius:6px;overflow:auto;font-size:12px;white-space:pre-wrap}
.llmr .better{color:var(--llmr-ok);font-weight:600}.llmr .worse{color:var(--llmr-err);font-weight:600}
@media (max-width:640px){.llmr .llmr-row{grid-template-columns:1fr auto}.llmr .llmr-row .llmr-text{grid-column:1/-1}}
"""


def _metrics_table(metrics: Dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{_esc(pretty_key(k))}</td><td class='num'>{_esc(format_value(k, v))}</td></tr>"
        for k, v in metrics.items()
    )
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>" if rows else ""


def _list_table(items: List[dict], limit: int = 20) -> str:
    if not items:
        return ""
    cols = list(items[0].keys())
    head = "".join(f"<th>{_esc(pretty_key(c))}</th>" for c in cols)
    body = ""
    for item in items[:limit]:
        cells = ""
        for c in cols:
            v = item.get(c, "")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cells += f"<td class='num'>{_esc(format_value(c, v))}</td>"
            else:
                cells += f"<td>{_esc(_truncate(v))}</td>"
        body += f"<tr>{cells}</tr>"
    more = f"<p class='llmr-notes'>Showing {limit} of {len(items)} rows.</p>" if len(items) > limit else ""
    return f"<table><tr>{head}</tr>{body}</table>{more}"


def _bars(values: Dict[str, float]) -> str:
    total = sum(values.values()) or 1
    rows = ""
    for k, v in sorted(values.items(), key=lambda kv: -kv[1]):
        pct = 100 * v / total
        rows += (
            f"<tr><td style='width:130px'>{_esc(pretty_key(k))}</td>"
            f"<td><div class='llmr-bar' style='width:{pct:.1f}%'></div></td>"
            f"<td class='num' style='width:120px'>{_esc(human_number(v))} ({pct:.1f}%)</td></tr>"
        )
    return f"<table>{rows}</table>"


def _details_html(result: AnalyzerResult) -> str:
    parts = []
    for key, value in result.details.items():
        if key == "traceback":
            parts.append(f"<h4>Traceback</h4><pre>{_esc(value)}</pre>")
        elif key == "param_breakdown" and isinstance(value, dict):
            parts.append(f"<h4>Where the parameters are</h4>{_bars(value)}")
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            parts.append(f"<h4>{_esc(pretty_key(key))}</h4>{_list_table(value)}")
        elif isinstance(value, dict) and value:
            flat = {k: v for k, v in value.items() if not isinstance(v, (dict, list))}
            nested = {k: v for k, v in value.items() if isinstance(v, (dict, list))}
            section = _metrics_table(flat).replace("<th>Metric</th><th>Value</th>", "<th>Field</th><th>Value</th>")
            for k, v in nested.items():
                section += f"<pre>{_esc(k)}: {_esc(json.dumps(v, indent=1, default=str))}</pre>"
            parts.append(f"<h4>{_esc(pretty_key(key))}</h4>{section}")
        elif value not in (None, "", [], {}):
            parts.append(f"<p><b>{_esc(pretty_key(key))}:</b> {_esc(value)}</p>")
    return "".join(parts)


# ----------------------------------------------------------------- Report


class Report:
    """Results of :func:`llmreport.analyze`.

    In a notebook, just put ``report`` on the last line of a cell to see it.
    Access one check with ``report["perplexity"]`` and all numbers with ``report.metrics``.
    """

    def __init__(self, results: List[AnalyzerResult], metadata: Optional[Dict[str, Any]] = None):
        self.results = list(results)
        self.metadata = dict(metadata or {})

    # -- access
    def __getitem__(self, name: str) -> AnalyzerResult:
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(f"No check named {name!r}. Available: {[r.name for r in self.results]}")

    def __iter__(self) -> Iterator[AnalyzerResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __contains__(self, name: str) -> bool:
        return any(r.name == name for r in self.results)

    @property
    def metrics(self) -> Dict[str, Any]:
        """All metrics in one flat dict, keyed ``"check.metric"``."""
        return {f"{r.name}.{k}": v for r in self.results for k, v in r.metrics.items()}

    @property
    def warnings(self) -> List[AnalyzerResult]:
        return [r for r in self.results if r.status in ("warning", "error")]

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model", "model"))

    # -- plain text
    def __str__(self) -> str:
        width = max([len(r.title) for r in self.results] + [10])
        lines = [f"llmreport: {self.model_name}", self._meta_line(), ""]
        for r in self.results:
            badge = f"[{STATUS_LABEL.get(r.status, r.status)}]"
            lines.append(f"{r.title:<{width}}  {badge:<9} {r.summary}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<llmreport.Report model={self.model_name!r} checks={[r.name for r in self.results]}>"

    def _meta_line(self) -> str:
        m = self.metadata
        bits = [m.get("created_utc"), f"mode={m.get('mode')}" if m.get("mode") else None,
                f"device={m.get('device')}" if m.get("device") else None,
                f"took {m['duration_s']}s" if "duration_s" in m else None]
        return " · ".join(b for b in bits if b)

    def show(self) -> None:
        """Pretty-print in a terminal (uses ``rich`` if installed)."""
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            print(self)
            return
        colors = {"ok": "green", "warning": "yellow", "error": "red", "info": "blue", "skipped": "blue"}
        table = Table(title=f"llmreport: {self.model_name}", caption=self._meta_line(), show_lines=True)
        table.add_column("Check", style="bold")
        table.add_column("Status")
        table.add_column("Summary")
        for r in self.results:
            c = colors.get(r.status, "white")
            table.add_row(r.title, f"[{c}]{STATUS_LABEL.get(r.status, r.status)}[/{c}]", r.summary)
        Console().print(table)

    # -- HTML
    def _html_fragment(self, standalone: bool = False, open_sections: bool = False) -> str:
        score = "".join(
            f"<div class='llmr-row'><div class='llmr-name'>{_esc(r.title)}</div>"
            f"<div><span class='llmr-badge s-{_esc(r.status)}'>{STATUS_ICON.get(r.status, '')} "
            f"{_esc(STATUS_LABEL.get(r.status, r.status))}</span></div>"
            f"<div class='llmr-text'>{_esc(r.summary)}</div></div>"
            for r in self.results
        )
        sections = ""
        for r in self.results:
            notes = "".join(f"<li>{_esc(n)}</li>" for n in r.notes)
            notes = f"<ul class='llmr-notes'>{notes}</ul>" if notes else ""
            sections += (
                f"<details{' open' if open_sections else ''}><summary>{_esc(r.title)} "
                f"<span class='llmr-badge s-{_esc(r.status)}'>{_esc(STATUS_LABEL.get(r.status, r.status))}</span>"
                f"<span style='color:var(--llmr-muted);font-weight:400;font-size:12px'> · {r.duration_s:.1f}s</span>"
                f"</summary><div class='llmr-body'><p class='llmr-sum'>{_esc(r.summary)}</p>"
                f"{_metrics_table(r.metrics)}{notes}{_details_html(r)}</div></details>"
            )
        m = self.metadata
        versions = " · ".join(
            f"{k.replace('_version', '')} {m[k]}" for k in
            ("llmreport_version", "torch_version", "transformers_version", "python_version") if k in m
        )
        cls = "llmr llmr-standalone" if standalone else "llmr"
        return (
            f"<style>{_CSS}</style><div class='{cls}'>"
            f"<h2>{_esc(self.model_name)}</h2>"
            f"<div class='llmr-meta'>{_esc(self._meta_line())}</div>"
            f"<div class='llmr-score'>{score}</div>{sections}"
            f"<div class='llmr-meta' style='margin-top:14px'>Generated by llmreport · {_esc(versions)}</div></div>"
        )

    def _repr_html_(self) -> str:
        return self._html_fragment()

    def to_html(self, path: Optional[str] = None) -> str:
        """Return a standalone HTML page, and write it to ``path`` if given."""
        page = (
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>llmreport · {_esc(self.model_name)}</title>"
            "<style>body{margin:0;padding:24px 16px;background:#ffffff}"
            "@media (prefers-color-scheme:dark){body{background:#0d1117}}"
            ".llmr{margin:0 auto}</style></head><body>"
            f"{self._html_fragment(standalone=True)}</body></html>"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(page)
        return page

    # -- Markdown / model card
    def to_markdown(self, path: Optional[str] = None, model_card: bool = True) -> str:
        """Return a Markdown report. With ``model_card=True`` it is laid out as a
        Hugging Face-style model card with placeholders for what only you know."""
        m = self.metadata
        out = []
        if model_card:
            library = "transformers" if m.get("transformers_model", True) else "pytorch"
            out += [
                "---",
                f"library_name: {library}",
                "tags:",
                "- llmreport",
                "---",
                "",
                f"# Model card: {self.model_name}",
                "",
                "## Model details",
                "",
                "- **Developed by:** _TODO_",
                "- **Model type:** " + str(self._arch_value("architecture_class")),
                "- **Language(s):** _TODO_",
                "- **License:** _TODO_",
                "- **Fine-tuned from:** _TODO (or remove)_",
                "",
                "## Intended use",
                "",
                "_TODO: what this model is for, and what it should not be used for._",
                "",
                "## Training data",
                "",
                "_TODO: describe the data the model was trained or fine-tuned on._",
                "",
                "## Evaluation (generated by llmreport)",
                "",
            ]
        else:
            out += [f"# llmreport: {self.model_name}", ""]
        out += [f"_{self._meta_line()}_", "", "| Check | Status | Summary |", "|---|---|---|"]
        for r in self.results:
            out.append(f"| {r.title} | {STATUS_LABEL.get(r.status, r.status)} | {r.summary.replace('|', '/')} |")
        out.append("")
        for r in self.results:
            out += [f"### {r.title}", "", r.summary, ""]
            if r.metrics:
                out += ["| Metric | Value |", "|---|---|"]
                out += [f"| {pretty_key(k)} | {format_value(k, v)} |" for k, v in r.metrics.items()]
                out.append("")
            out += [f"- {n}" for n in r.notes]
            if r.notes:
                out.append("")
        if model_card:
            out += [
                "## Limitations and biases",
                "",
                "_TODO: known failure modes. The llmreport checks above are quick indicators "
                "on small prompt sets, not a complete evaluation._",
                "",
            ]
        versions = ", ".join(f"{k.replace('_version', '')} {m[k]}" for k in
                             ("llmreport_version", "torch_version", "transformers_version") if k in m)
        out.append(f"<sub>Generated with llmreport ({versions}).</sub>")
        text = "\n".join(out) + "\n"
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def _arch_value(self, key: str) -> Any:
        if "architecture" in self:
            return self["architecture"].details.get(key, "_TODO_")
        return "_TODO_"

    # -- JSON
    def to_dict(self) -> Dict[str, Any]:
        return {"metadata": self.metadata, "results": [r.to_dict() for r in self.results]}

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Report":
        return cls([AnalyzerResult(**r) for r in data["results"]], data.get("metadata", {}))

    @classmethod
    def load(cls, path: str) -> "Report":
        """Load a report saved with :meth:`to_json`."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # -- comparison
    def compare(self, other: "Report", names: Optional[tuple] = None) -> "Comparison":
        """Compare numeric metrics with another report, for example before vs after fine-tuning."""
        return Comparison(self, other, names)


def _meaningful_change(metric: str, a: float, b: float) -> bool:
    """Small differences are noise, not "better" or "worse"."""
    delta = abs(b - a)
    if format_value(metric, 0.5).endswith("%"):  # fractions shown as percentages
        return delta >= 0.01  # at least one percentage point
    if metric.startswith(("ttft", "decode_tokens")):  # timings vary run to run
        return delta > 0.10 * max(abs(a), abs(b), 1e-9)
    return delta > 0.02 * max(abs(a), abs(b), 1e-9)


class Comparison:
    """Side-by-side view of two reports. Displays as a table in notebooks."""

    def __init__(self, a: Report, b: Report, names: Optional[tuple] = None):
        self.a, self.b = a, b
        self.names = names or (a.model_name, b.model_name)
        if self.names[0] == self.names[1]:
            self.names = (f"{self.names[0]} (A)", f"{self.names[1]} (B)")
        self.rows = self._build()

    def _build(self) -> List[Dict[str, Any]]:
        ma, mb = self.a.metrics, self.b.metrics
        rows = []
        for key in list(ma) + [k for k in mb if k not in ma]:
            va, vb = ma.get(key), mb.get(key)
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (va, vb)):
                continue
            metric = key.split(".", 1)[1]
            direction = METRIC_DIRECTION.get(metric, 0)
            delta = vb - va
            verdict = ""
            meaningful = _meaningful_change(metric, va, vb)
            if direction and meaningful and not (math.isnan(va) or math.isnan(vb)):
                verdict = "better" if delta * direction > 0 else "worse"
            rows.append({"metric": key, "a": va, "b": vb, "change": delta, "verdict": verdict})
        return rows

    def __str__(self) -> str:
        lines = [f"{'Metric':<45} {self.names[0][:18]:>18} {self.names[1][:18]:>18}  Change"]
        for r in self.rows:
            m = r["metric"].split(".", 1)[1]
            lines.append(
                f"{r['metric']:<45} {format_value(m, r['a']):>18} {format_value(m, r['b']):>18}  {r['verdict']}"
            )
        return "\n".join(lines)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        body = ""
        for r in self.rows:
            m = r["metric"].split(".", 1)[1]
            cls = r["verdict"]
            body += (
                f"<tr><td>{_esc(r['metric'])}</td><td class='num'>{_esc(format_value(m, r['a']))}</td>"
                f"<td class='num'>{_esc(format_value(m, r['b']))}</td>"
                f"<td class='{cls}'>{_esc(cls)}</td></tr>"
            )
        return (
            f"<style>{_CSS}</style><div class='llmr'><h2>Comparison</h2><table><tr><th>Metric</th>"
            f"<th>{_esc(self.names[0])}</th><th>{_esc(self.names[1])}</th><th></th></tr>{body}</table></div>"
        )
