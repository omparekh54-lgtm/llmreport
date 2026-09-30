"""Live training tracking.

Add two lines to any training loop::

    tracker = llmreport.Tracker(model, tokenizer, log_dir="runs/pretrain", eval_texts=val_texts)
    for step, (x, y) in enumerate(loader):
        loss = model(x, y)
        loss.backward()
        tracker.step(loss, optimizer=optimizer, tokens=x.numel())   # after backward, before zero_grad
        optimizer.step(); optimizer.zero_grad()
    tracker.close()

Every step records the loss, learning rate, gradient norm, speed and memory. Every
``eval_every`` steps it measures perplexity on your texts, writes sample generations
and checks the weights' health. Problems (NaN loss, loss spikes, exploding or
vanishing gradients, plateaus, overfitting, broken weights) are reported the moment
they happen. Everything is written to ``log_dir``: ``metrics.jsonl``, ``events.jsonl``
and a ``dashboard.html`` that refreshes itself, so you can watch from a browser or
from another terminal with ``llmreport watch runs/pretrain``.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import sys
import time
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch

__all__ = ["Tracker", "Run", "load_run", "track_checkpoints"]

CRITICAL, WARNING, INFO = "critical", "warning", "info"
_WARMUP = 50  # steps of history before spikes are judged
_ICON = {CRITICAL: "✕", WARNING: "!", INFO: "i"}


def _float(x) -> Optional[float]:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach()
        if x.numel() != 1:
            x = x.float().mean()
        x = x.item()
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and type(shell).__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line being written right now
    return rows


class Tracker:
    """Track a training run live. See the module docstring for a full example.

    Args:
        model: The model being trained (optional). Needed for gradient norms, evaluation,
            samples and health checks.
        tokenizer: Its tokenizer (any kind llmreport supports). Needed for evaluation and samples.
        log_dir: Folder for ``metrics.jsonl``, ``events.jsonl``, ``meta.json`` and ``dashboard.html``.
            If it already holds a run, tracking continues where it stopped (resumed training).
        name: Display name of the run.
        eval_texts: Texts to measure perplexity on (for example held-out validation samples).
        eval_every: Evaluate every N steps (perplexity, samples, health). Default: 500 when there is
            something to evaluate.
        sample_prompts: Prompts to generate from at each evaluation, so you can read how the model is
            doing (for example ``["def fibonacci(n):"]``).
        prompt_template: Template for the sample prompts, e.g. ``"<|instruction|>{prompt}<|response|>"``.
        sample_tokens: Tokens to generate per sample.
        health_every: Check weight health every N steps (default: with each evaluation).
        print_every: Print a progress line every N steps (0 to stay quiet). In Jupyter a live panel is
            shown instead.
        total_steps: Planned number of steps, to show progress and time remaining.
        optimizer: Optimizer to read the learning rate from (you can also pass it to ``step``).
        dashboard: Write ``dashboard.html`` (refreshed every ``dashboard_every`` seconds).
        alerts: Detect problems and report them.
        grad_every: Compute the gradient norm every N steps (it walks all parameters).
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        *,
        log_dir: str = "llmreport_runs/run",
        name: Optional[str] = None,
        eval_texts: Optional[Sequence[str]] = None,
        eval_every: Optional[int] = None,
        sample_prompts: Optional[Sequence[str]] = None,
        prompt_template: Optional[str] = None,
        sample_tokens: int = 48,
        health_every: Optional[int] = None,
        print_every: int = 10,
        total_steps: Optional[int] = None,
        optimizer=None,
        dashboard: bool = True,
        dashboard_every: float = 20.0,
        alerts: bool = True,
        grad_every: int = 1,
        notebook: Union[str, bool] = "auto",
        on_alert: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.name = name or os.path.basename(os.path.normpath(log_dir))
        self.eval_texts = [eval_texts] if isinstance(eval_texts, str) else list(eval_texts or [])
        self.sample_prompts = [sample_prompts] if isinstance(sample_prompts, str) else list(sample_prompts or [])
        can_eval = model is not None and tokenizer is not None and (self.eval_texts or self.sample_prompts)
        self.eval_every = eval_every if eval_every is not None else (500 if can_eval else 0)
        self.health_every = health_every if health_every is not None else (self.eval_every or 0)
        self.prompt_template = prompt_template
        self.sample_tokens = sample_tokens
        self.print_every = print_every
        self.total_steps = total_steps
        self.optimizer = optimizer
        self.dashboard = dashboard
        self.dashboard_every = dashboard_every
        self.alerts_enabled = alerts
        self.grad_every = max(1, grad_every)
        self.on_alert = on_alert
        self.notebook = _in_notebook() if notebook == "auto" else bool(notebook)

        os.makedirs(log_dir, exist_ok=True)
        self.metrics_path = os.path.join(log_dir, "metrics.jsonl")
        self.events_path = os.path.join(log_dir, "events.jsonl")
        self.meta_path = os.path.join(log_dir, "meta.json")
        self.dashboard_path = os.path.join(log_dir, "dashboard.html")

        self.rows: List[Dict[str, Any]] = _read_jsonl(self.metrics_path)
        self.events: List[Dict[str, Any]] = _read_jsonl(self.events_path)
        self.resumed = bool(self.rows)
        self._step = int(self.rows[-1]["step"]) if self.rows else 0
        self._ema: Optional[float] = None
        self._ema_n = 0
        for r in self.rows[-200:]:
            if r.get("loss") is not None and math.isfinite(r["loss"]):
                self._update_ema(r["loss"])
        self._losses = deque((r["loss"] for r in self.rows[-200:] if r.get("loss") is not None), maxlen=200)
        self._grads = deque((r["grad_norm"] for r in self.rows[-200:] if r.get("grad_norm") is not None),
                            maxlen=200)
        self._tps = deque((r["tokens_per_s"] for r in self.rows[-50:] if r.get("tokens_per_s")), maxlen=50)
        self._last_alert: Dict[str, int] = {}
        self._tiny_grads = 0
        self._last_time: Optional[float] = None
        self._start_time = time.time()
        self._last_dash = 0.0
        self._display = None
        self._metrics_file = open(self.metrics_path, "a", encoding="utf-8")
        self._events_file = open(self.events_path, "a", encoding="utf-8")
        self.closed = False
        self._write_meta()
        self._event("start" if not self.resumed else "resume", step=self._step,
                    message=f"Tracking {'resumed' if self.resumed else 'started'} at step {self._step}.")

    # ------------------------------------------------------------------ setup
    def _write_meta(self) -> None:
        meta: Dict[str, Any] = {"name": self.name, "total_steps": self.total_steps,
                                "eval_every": self.eval_every, "prompt_template": self.prompt_template}
        if self.model is not None:
            try:
                from .functions import info

                was = getattr(self.model, "training", False)
                details = info(self.model, self.tokenizer)
                if hasattr(self.model, "train"):
                    self.model.train(was)
                meta["model"] = {k: v for k, v in details.items()
                                 if isinstance(v, (int, float, str, bool)) or v is None}
            except Exception as exc:  # never break training because of a report
                meta["model"] = {"name": type(self.model).__name__, "error": f"{type(exc).__name__}: {exc}"}
        old = {}
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, encoding="utf-8") as f:
                    old = json.load(f)
            except Exception:
                old = {}
        meta["created"] = old.get("created") or time.strftime("%Y-%m-%d %H:%M:%S")
        meta["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, default=str)
        self.meta = meta

    # ------------------------------------------------------------------ recording
    def _update_ema(self, loss: float, beta: float = 0.98) -> None:
        self._ema_n += 1
        self._ema = loss if self._ema is None else beta * self._ema + (1 - beta) * loss

    def step(self, loss=None, *, step: Optional[int] = None, lr: Optional[float] = None,
             tokens: Optional[int] = None, samples: Optional[int] = None, optimizer=None,
             grad_norm: Optional[float] = None, **extra) -> Dict[str, Any]:
        """Record one training step. Call it after ``loss.backward()`` and before ``optimizer.zero_grad()``
        so the gradient norm can be measured.

        Args:
            loss: The step's loss (tensor or number).
            step: Global step number (default: previous step + 1).
            lr: Learning rate (default: read from the optimizer).
            tokens: Tokens processed in this step, for tokens/second.
            samples: Sequences processed in this step, for samples/second.
            optimizer: Optimizer to read the learning rate from.
            grad_norm: Gradient norm if you already computed it (e.g. from ``clip_grad_norm_``).
            **extra: Any other numbers to record and chart (for example ``aux_loss=...``).
        """
        now = time.time()
        self._step = int(step) if step is not None else self._step + 1
        row: Dict[str, Any] = {"step": self._step, "time": round(now, 3)}
        value = _float(loss)
        if value is not None:
            row["loss"] = value
            if math.isfinite(value):
                self._update_ema(value)
                row["loss_avg"] = self._ema
        opt = optimizer or self.optimizer
        lr_value = _float(lr) if lr is not None else None
        if lr_value is None and opt is not None:
            groups = getattr(opt, "param_groups", None)
            if groups:
                lr_value = _float(groups[0].get("lr"))
        if lr_value is not None:
            row["lr"] = lr_value
        g = _float(grad_norm)
        if g is None and self.model is not None and self._step % self.grad_every == 0:
            from .healthcheck import grad_norm as _gn

            g = _gn(self.model)
        if g is not None:
            row["grad_norm"] = g
        if self._last_time is not None:
            dt = now - self._last_time
            row["step_time"] = dt
            if tokens and dt > 0:
                row["tokens_per_s"] = tokens / dt
            if samples and dt > 0:
                row["samples_per_s"] = samples / dt
        if tokens:
            row["tokens"] = int(tokens)
        if torch.cuda.is_available():
            try:
                row["gpu_mem_bytes"] = torch.cuda.memory_allocated()
                row["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
            except Exception:
                pass
        for k, v in extra.items():
            fv = _float(v)
            if fv is not None:
                row[k] = fv

        self.rows.append(row)
        self._metrics_file.write(json.dumps(row) + "\n")
        self._metrics_file.flush()
        if self.alerts_enabled:
            self._check_step(row)
        if value is not None:
            self._losses.append(value)
        if g is not None:
            self._grads.append(g)
        if row.get("tokens_per_s"):
            self._tps.append(row["tokens_per_s"])

        if self.eval_every and self._step % self.eval_every == 0:
            self.evaluate()
        elif self.health_every and self._step % self.health_every == 0:
            self.check_health()
        if self.print_every and self._step % self.print_every == 0:
            self._print(row)
        if self.dashboard and now - self._last_dash >= self.dashboard_every:
            self.write_dashboard()
        # Restart the clock after llmreport's own work (evaluation, dashboard), so step time and speed
        # measure only your training code.
        self._last_time = time.time()
        return row

    def log(self, step: Optional[int] = None, **metrics) -> Dict[str, Any]:
        """Record evaluation numbers you computed yourself, e.g. ``tracker.log(val_loss=1.23)``."""
        values = {k: _float(v) for k, v in metrics.items() if _float(v) is not None}
        event = self._event("eval", step=step if step is not None else self._step, **values)
        if self.alerts_enabled:
            self._check_eval(event)
        return event

    # ------------------------------------------------------------------ evaluation
    def evaluate(self) -> Dict[str, Any]:
        """Measure perplexity on ``eval_texts``, generate ``sample_prompts`` and check weight health, now."""
        if self.model is None or self.tokenizer is None:
            return {}
        from .functions import generate, perplexity, text_stats

        was = getattr(self.model, "training", False)
        out: Dict[str, Any] = {}
        t0 = time.time()
        try:
            if self.eval_texts:
                res = perplexity(self.model, self.tokenizer, self.eval_texts, details=True)
                out["eval_perplexity"] = res["perplexity"]
                out["eval_loss"] = res["mean_nll"]
                out["eval_bits_per_char"] = res["bits_per_char"]
            samples = []
            for prompt in self.sample_prompts:
                text = generate(self.model, self.tokenizer, prompt, self.sample_tokens, template=self.prompt_template)
                samples.append({"prompt": prompt, "output": text})
            if samples:
                out["sample_repetition"] = text_stats([s["output"] for s in samples])["repeated_3gram_rate"]
        except Exception as exc:
            self._alert(WARNING, "eval_error", f"Evaluation failed: {type(exc).__name__}: {exc}")
            samples = []
        finally:
            if hasattr(self.model, "train"):
                self.model.train(was)
        out["eval_seconds"] = time.time() - t0
        event = self._event("eval", step=self._step, **out)
        if samples:
            self._event("samples", step=self._step, samples=samples)
        if self.alerts_enabled:
            self._check_eval(event)
        self.check_health()
        if self.dashboard:
            self.write_dashboard()
        return out

    def check_health(self):
        """Run :func:`llmreport.health` on the model now and report anything wrong."""
        if self.model is None:
            return None
        from .healthcheck import health

        try:
            report = health(self.model)
        except Exception as exc:
            self._alert(INFO, "health_error", f"Health check failed: {type(exc).__name__}: {exc}")
            return None
        self._event("health", step=self._step, status=report.status, summary=report.summary,
                    issues=[str(i) for i in report.issues])
        for issue in report.issues:
            if issue.severity in (CRITICAL, WARNING):
                self._alert(issue.severity, "health:" + issue.problem[:40], f"Weights: {issue}")
        return report

    # ------------------------------------------------------------------ alerts
    def _event(self, kind: str, step: Optional[int] = None, **data) -> Dict[str, Any]:
        event = {"kind": kind, "step": step if step is not None else self._step, "time": round(time.time(), 3),
                 **data}
        self.events.append(event)
        if not self.closed:
            self._events_file.write(json.dumps(event, default=str) + "\n")
            self._events_file.flush()
        return event

    def _alert(self, severity: str, key: str, message: str, cooldown: int = 50) -> None:
        last = self._last_alert.get(key)
        if last is not None and self._step - last < cooldown:
            return
        self._last_alert[key] = self._step
        event = self._event("alert", step=self._step, severity=severity, key=key, message=message)
        line = f"llmreport [{_ICON[severity]} {severity}] step {self._step}: {message}"
        print(line, file=sys.stderr if severity == CRITICAL else sys.stdout, flush=True)
        if self.on_alert is not None:
            try:
                self.on_alert(event)
            except Exception:
                pass

    @property
    def alerts(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == "alert"]

    def _check_step(self, row: Dict[str, Any]) -> None:
        loss = row.get("loss")
        if loss is not None and not math.isfinite(loss):
            self._alert(CRITICAL, "loss_nan", f"Loss is {loss}. Training has diverged: stop, go back to the last "
                        "good checkpoint, and lower the learning rate or add gradient clipping.", cooldown=10)
        elif loss is not None and len(self._losses) >= _WARMUP:
            recent = [x for x in list(self._losses)[-50:] if math.isfinite(x)]
            if len(recent) >= _WARMUP:
                med = statistics.median(recent)
                mad = statistics.median(abs(x - med) for x in recent)
                if loss > med + max(6 * mad, 0.25 * abs(med)):
                    self._alert(WARNING, "loss_spike", f"Loss spike: {loss:.4g} against a recent median of "
                                f"{med:.4g}. One spike is often harmless; repeated spikes mean the learning rate "
                                "is too high or a bad batch.")
        g = row.get("grad_norm")
        if g is not None:
            if not math.isfinite(g):
                self._alert(CRITICAL, "grad_nan", "Gradients are NaN or infinite. The optimizer step will corrupt "
                            "the weights; skip it and lower the learning rate or check for float16 overflow.",
                            cooldown=10)
            else:
                recent = [x for x in list(self._grads)[-50:] if math.isfinite(x)]
                if len(recent) >= _WARMUP:  # the first steps are naturally unstable
                    med = statistics.median(recent)
                    if med > 0 and g > 10 * med:
                        self._alert(WARNING, "grad_spike", f"Gradient norm jumped to {g:.4g} (recent median "
                                    f"{med:.4g}). Gradient clipping (e.g. clip_grad_norm_(..., 1.0)) prevents "
                                    "this from damaging the weights.")
                self._tiny_grads = self._tiny_grads + 1 if g < 1e-7 else 0
                if self._tiny_grads == 3:
                    self._alert(WARNING, "grad_vanish", "Gradient norm has been almost zero for 3 steps: the "
                                "model isn't learning (vanishing gradients, frozen layers, or zero learning rate).")
        # Plateau: the smoothed loss hasn't improved over the last window.
        window = max(200, len(self.rows) // 5)
        if len(self.rows) >= 2 * window and len(self.rows) % 50 == 0:
            avgs = [r["loss_avg"] for r in self.rows if r.get("loss_avg") is not None]
            if len(avgs) >= 2 * window:
                before = min(avgs[:-window])
                recent = min(avgs[-window:])
                if recent > before - 0.005 * abs(before):
                    self._alert(INFO, "plateau", f"Loss hasn't improved in the last {window} steps (best "
                                f"{before:.4g} before, {recent:.4g} since). Consider lowering the learning rate, "
                                "more data, or stopping.", cooldown=window)
        tps = row.get("tokens_per_s")
        if tps and len(self._tps) >= 30:
            history = list(self._tps)
            med = statistics.median(history[:-5])
            tps = statistics.median(history[-4:] + [tps])  # sustained over 5 steps, not one slow step
            if tps < 0.4 * med:
                self._alert(INFO, "slow", f"Speed dropped to {tps:,.0f} tokens/s (usual {med:,.0f}). Another "
                            "program, thermal throttling, or a slow evaluation may be the cause.", cooldown=200)

    def _check_eval(self, event: Dict[str, Any]) -> None:
        ppl = event.get("eval_perplexity")
        if ppl is not None and not math.isfinite(ppl):
            self._alert(CRITICAL, "eval_nan", "Evaluation perplexity is NaN or infinite.")
        key = next((k for k in ("eval_perplexity", "eval_loss", "val_loss", "val_perplexity") if k in event), None)
        evals = [e for e in self.events if e.get("kind") == "eval" and e.get(key) is not None] if key else []
        if len(evals) >= 3:
            a, b, c = (e[key] for e in evals[-3:])
            train = [r["loss_avg"] for r in self.rows if r.get("loss_avg") is not None]
            gap = 2 * max(1, self.eval_every or (evals[-1]["step"] - evals[-3]["step"]) // 2)
            train_falling = len(train) > 10 and train[-1] < train[max(0, len(train) - 1 - gap)]
            if a < b < c and train_falling:
                label = {"eval_perplexity": "perplexity", "eval_loss": "loss", "val_loss": "validation loss",
                         "val_perplexity": "validation perplexity"}[key]
                self._alert(WARNING, "overfit", f"Evaluation {label} rose twice in a row ({a:.4g} → {b:.4g} → "
                            f"{c:.4g}) while training loss keeps falling: the model may be overfitting. Keep the "
                            "best checkpoint and consider stopping or adding data/regularisation.",
                            cooldown=2 * max(1, self.eval_every or 1))
        rep = event.get("sample_repetition")
        if rep is not None and rep > 0.5:
            self._alert(INFO, "sample_loops", f"Sample generations are mostly repeated phrases ({rep:.0%}). Normal "
                        "early in training; worrying late.", cooldown=5 * max(1, self.eval_every))

    # ------------------------------------------------------------------ output
    def _progress(self, row) -> str:
        parts = [f"step {row['step']:>7,}"]
        if self.total_steps:
            parts[0] += f"/{self.total_steps:,}"
        if row.get("loss") is not None:
            parts.append(f"loss {row['loss']:.4f}" + (f" (avg {row['loss_avg']:.4f})" if row.get("loss_avg") else ""))
        if row.get("lr") is not None:
            parts.append(f"lr {row['lr']:.2e}")
        if row.get("grad_norm") is not None:
            parts.append(f"grad {row['grad_norm']:.3g}")
        if row.get("tokens_per_s"):
            parts.append(f"{row['tokens_per_s']:,.0f} tok/s")
        if row.get("step_time"):
            parts.append(f"{row['step_time']:.2f} s/step")
        if row.get("gpu_mem_bytes"):
            parts.append(f"GPU {row['gpu_mem_bytes'] / 1e9:.1f} GB")
        eta = self.eta_seconds()
        if eta is not None:
            parts.append(f"ETA {_fmt_duration(eta)}")
        return " | ".join(parts)

    def eta_seconds(self) -> Optional[float]:
        if not self.total_steps:
            return None
        times = [r["step_time"] for r in self.rows[-50:] if r.get("step_time")]
        if not times:
            return None
        return max(0, self.total_steps - self._step) * statistics.median(times)

    def _print(self, row) -> None:
        if self.notebook:
            self._update_notebook()
        else:
            print("llmreport " + self._progress(row), flush=True)

    def _update_notebook(self) -> None:
        try:
            from IPython.display import HTML, display

            from .dashboard import render_panel

            panel = HTML(render_panel(self))
            if self._display is None:
                self._display = display(panel, display_id=True)
            else:
                self._display.update(panel)
        except Exception:
            self.notebook = False

    def write_dashboard(self, path: Optional[str] = None) -> str:
        """Write the HTML dashboard (default: ``log_dir/dashboard.html``) and return its path."""
        from .dashboard import render_dashboard

        path = path or self.dashboard_path
        page = render_dashboard(self.meta, self.rows, self.events, live=not self.closed)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(page)
        os.replace(tmp, path)  # the browser never sees a half-written page
        self._last_dash = time.time()
        return path

    def summary(self) -> Dict[str, Any]:
        """Key numbers for the run so far."""
        return run_summary(self.rows, self.events, self.total_steps)

    def show(self):
        """Show the live panel in a notebook."""
        self._update_notebook()

    def close(self) -> None:
        """Finish: write the final dashboard and close the log files."""
        if self.closed:
            return
        self._event("end", step=self._step, message=f"Tracking ended at step {self._step}.")
        self.closed = True
        self._metrics_file.close()
        self._events_file.close()
        if self.dashboard:
            self.write_dashboard()
        if self.notebook:
            self._update_notebook()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            if not self.closed:
                self._metrics_file.flush()
                self._events_file.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------ integrations
    def hf_callback(self):
        """A ``transformers.TrainerCallback`` that feeds this tracker: ``Trainer(..., callbacks=[tracker.hf_callback()])``."""
        from transformers import TrainerCallback

        tracker = self

        class LLMReportCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                if "loss" in logs:
                    tracker.step(logs["loss"], step=state.global_step, lr=logs.get("learning_rate"),
                                 grad_norm=logs.get("grad_norm"))
                evals = {k: v for k, v in logs.items() if k.startswith("eval_")}
                if evals:
                    tracker.log(step=state.global_step, **evals)

            def on_train_end(self, args, state, control, **kwargs):
                tracker.close()

        return LLMReportCallback()


# ====================================================================== reading runs


def run_summary(rows, events, total_steps=None) -> Dict[str, Any]:
    losses = [(r["step"], r["loss"]) for r in rows if r.get("loss") is not None and math.isfinite(r["loss"])]
    out: Dict[str, Any] = {"steps": rows[-1]["step"] if rows else 0, "rows": len(rows)}
    if total_steps:
        out["total_steps"] = total_steps
        out["progress"] = out["steps"] / total_steps
    if losses:
        best = min(losses, key=lambda x: x[1])
        out.update(first_loss=losses[0][1], last_loss=losses[-1][1], best_loss=best[1], best_loss_step=best[0])
        avgs = [r["loss_avg"] for r in rows if r.get("loss_avg") is not None]
        if avgs:
            out["last_loss_avg"] = avgs[-1]
    tps = [r["tokens_per_s"] for r in rows if r.get("tokens_per_s")]
    if tps:
        out["tokens_per_s"] = statistics.median(tps[-100:])
    tokens = sum(r.get("tokens", 0) for r in rows)
    if tokens:
        out["tokens_seen"] = tokens
    times = [r["time"] for r in rows if r.get("time")]
    if len(times) > 1:
        out["elapsed_s"] = times[-1] - times[0]
    evals = [e for e in events if e.get("kind") == "eval"]
    if evals:
        out["last_eval"] = {k: v for k, v in evals[-1].items() if k not in ("kind", "time")}
        ppl = [(e["step"], e["eval_perplexity"]) for e in evals if e.get("eval_perplexity") is not None
               and math.isfinite(e["eval_perplexity"])]
        if ppl:
            best = min(ppl, key=lambda x: x[1])
            out["best_eval_perplexity"], out["best_eval_step"] = best[1], best[0]
    alerts = [e for e in events if e.get("kind") == "alert"]
    out["alerts"] = {s: sum(a.get("severity") == s for a in alerts) for s in (CRITICAL, WARNING, INFO)}
    return out


class Run:
    """A finished or running training run read back from its ``log_dir``."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.rows = _read_jsonl(os.path.join(log_dir, "metrics.jsonl"))
        self.events = _read_jsonl(os.path.join(log_dir, "events.jsonl"))
        meta_path = os.path.join(log_dir, "meta.json")
        self.meta: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)
        if not self.rows and not self.events:
            raise FileNotFoundError(f"No llmreport run found in {log_dir} (expected metrics.jsonl).")

    @property
    def alerts(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == "alert"]

    @property
    def evals(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == "eval"]

    def summary(self) -> Dict[str, Any]:
        return run_summary(self.rows, self.events, self.meta.get("total_steps"))

    def series(self, key: str) -> List[tuple]:
        """[(step, value), ...] for a metric from the step rows or the evaluations."""
        source = self.rows if any(key in r for r in self.rows) else self.evals
        return [(r["step"], r[key]) for r in source if r.get(key) is not None]

    def write_dashboard(self, path: Optional[str] = None) -> str:
        from .dashboard import render_dashboard

        path = path or os.path.join(self.log_dir, "dashboard.html")
        ended = any(e.get("kind") == "end" for e in self.events[-3:])
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_dashboard(self.meta, self.rows, self.events, live=not ended))
        return path

    def __repr__(self) -> str:
        s = self.summary()
        return f"<llmreport.Run {self.log_dir!r} steps={s['steps']} alerts={s['alerts']}>"


def load_run(log_dir: str) -> Run:
    """Read a run written by :class:`Tracker` (while it is still training, or after)."""
    return Run(log_dir)


# ====================================================================== checkpoints over time

_STEP_IN_NAME = re.compile(r"(\d+)(?!.*\d)")


def _checkpoint_step(path: str, obj: Any = None) -> Optional[int]:
    if isinstance(obj, dict):
        for key in ("step", "global_step", "iter_num", "iteration", "epoch"):
            if isinstance(obj.get(key), int):
                return obj[key]
    match = _STEP_IN_NAME.search(os.path.basename(path))
    return int(match.group(1)) if match else None


def track_checkpoints(
    checkpoints: Union[str, Iterable[str]],
    model_class=None,
    tokenizer=None,
    *,
    texts: Optional[Sequence[str]] = None,
    sample_prompts: Optional[Sequence[str]] = None,
    prompt_template: Optional[str] = None,
    code: bool = False,
    log_dir: Optional[str] = None,
    device: str = "auto",
    verbose: bool = True,
    **load_kwargs,
) -> Run:
    """Evaluate a series of saved checkpoints and chart how the model improved over training.

    Args:
        checkpoints: A folder (every ``.pt``/``.pth``/``.bin``/``.ckpt``/``.safetensors`` file in it, or
            subfolders with a Hugging Face model) or a list of paths.
        model_class: Your model's class, for weights-only checkpoints.
        tokenizer: Tokenizer object or path.
        texts: Validation texts for perplexity (default: built-in English passages).
        sample_prompts: Prompts to generate from at each checkpoint.
        code: Also run the Python code check (slower).
        log_dir: Where to write the run (default: ``<folder>/llmreport_progress``).

    Returns:
        A :class:`Run`; its ``dashboard.html`` charts perplexity (and code pass rate) against the step.

    Example:
        >>> llmreport.track_checkpoints("checkpoints/pretrain", GPT, "export/tokenizer.json",
        ...                             texts=val_texts, sample_prompts=["def add(a, b):"])
    """
    from .loading import CHECKPOINT_SUFFIXES, load_checkpoint

    if isinstance(checkpoints, (str, os.PathLike)):
        folder = os.fspath(checkpoints)
        entries = sorted(os.listdir(folder))
        paths = [os.path.join(folder, e) for e in entries
                 if e.endswith(CHECKPOINT_SUFFIXES) or os.path.isfile(os.path.join(folder, e, "config.json"))]
        log_dir = log_dir or os.path.join(folder, "llmreport_progress")
    else:
        paths = [os.fspath(p) for p in checkpoints]
        log_dir = log_dir or "llmreport_runs/checkpoints"
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints}")
    # Numbered checkpoints in step order; unnumbered ones (final.pt) last.
    ordered = sorted(paths, key=lambda p: (_checkpoint_step(p) is None, _checkpoint_step(p) or 0, p))

    tracker = Tracker(None, None, log_dir=log_dir, name=os.path.basename(os.path.normpath(log_dir)),
                      print_every=0, dashboard=False, alerts=True)
    tracker.eval_every = 1
    from .functions import code_score, generate, perplexity

    for i, path in enumerate(ordered):
        model, tok = load_checkpoint(path, model_class, tokenizer=tokenizer, device=device, **load_kwargs)
        step = _checkpoint_step(path)
        if step is None and os.path.isfile(path) and not path.endswith(".safetensors"):
            try:  # no number in the file name (e.g. final.pt): look for a "step" entry inside
                raw = torch.load(path, map_location="cpu", weights_only=False)
                step = _checkpoint_step(path, raw)
                del raw
            except Exception:
                step = None
        step = step if step is not None else i
        tracker._step = step
        values: Dict[str, Any] = {"checkpoint": os.path.basename(path)}
        res = perplexity(model, tok, texts, details=True)
        values.update(eval_perplexity=res["perplexity"], eval_loss=res["mean_nll"],
                      eval_bits_per_char=res["bits_per_char"])
        if code:
            values["code_pass_rate"] = code_score(model, tok, template=prompt_template).get("pass_rate")
        event = tracker._event("eval", step=step, **values)
        samples = [{"prompt": p, "output": generate(model, tok, p, 48, template=prompt_template)}
                   for p in (sample_prompts or [])]
        if samples:
            tracker._event("samples", step=step, samples=samples)
        tracker.model = model
        tracker.check_health()
        tracker.model = None
        tracker._check_eval(event)
        if verbose:
            extra = f", code pass rate {values['code_pass_rate']:.0%}" if code and values.get("code_pass_rate") \
                is not None else ""
            print(f"llmreport [{i + 1}/{len(ordered)}] {os.path.basename(path)} (step {step}): perplexity "
                  f"{res['perplexity']:,.2f}{extra}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    tracker.close()
    run = Run(log_dir)
    run.write_dashboard()
    return run
