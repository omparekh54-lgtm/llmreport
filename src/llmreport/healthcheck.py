"""Weight and gradient health checks: find broken layers without running the model.

``llmreport.health(model)`` looks at every weight (and gradient, if a backward pass
has run) and reports problems in plain English: NaN or infinite values, layers that
are all zeros, collapsed normalisation layers, dead neurons, extreme values,
outlier layers, and parameters that never receive a gradient.
"""

from __future__ import annotations

import html
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .structure import is_norm

CRITICAL, WARNING, INFO = "critical", "warning", "info"
_RANK = {CRITICAL: 0, WARNING: 1, INFO: 2}


@dataclass
class Issue:
    """One problem found by :func:`health`."""

    severity: str  # "critical", "warning" or "info"
    problem: str
    where: List[str] = field(default_factory=list)
    advice: str = ""

    def __str__(self) -> str:
        where = ""
        if self.where:
            shown = ", ".join(self.where[:4]) + (f" and {len(self.where) - 4} more" if len(self.where) > 4 else "")
            where = f" [{shown}]"
        return f"{self.severity.upper()}: {self.problem}{where}" + (f" {self.advice}" if self.advice else "")


@dataclass
class HealthReport:
    """Result of :func:`health`. ``report.ok`` is True when nothing critical or worrying was found."""

    status: str
    issues: List[Issue]
    layers: List[Dict[str, Any]]
    totals: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def summary(self) -> str:
        if not self.issues:
            n = self.totals.get("tensors", 0)
            return f"Healthy: no problems found in {n:,} weight tensors."
        counts = {s: sum(i.severity == s for i in self.issues) for s in (CRITICAL, WARNING, INFO)}
        parts = [f"{n} {s}" for s, n in counts.items() if n]
        return f"{self.status.capitalize()}: " + ", ".join(parts) + ". " + self.issues[0].problem

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "summary": self.summary, "issues": [asdict(i) for i in self.issues],
                "totals": self.totals, "layers": self.layers}

    def __repr__(self) -> str:
        lines = [f"llmreport health: {self.summary}"]
        lines += [f"  - {issue}" for issue in self.issues]
        t = self.totals
        lines.append(f"  weights: {t['tensors']:,} tensors, {t['params']:,} values, max |w| {t['abs_max']:.4g}")
        if t.get("grad_norm") is not None:
            lines.append(f"  gradients: total norm {t['grad_norm']:.4g}, {t['with_grad']:,} tensors have gradients")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        color = {"ok": "#1a7f37", "warning": "#9a6700", "critical": "#cf222e"}.get(self.status, "#57606a")
        icon = {"ok": "✓", "warning": "!", "critical": "✕"}.get(self.status, "i")
        items = "".join(
            f"<li><b>{html.escape(i.severity)}</b>: {html.escape(i.problem)}"
            + (f" <code>{html.escape(', '.join(i.where[:4]))}</code>" if i.where else "")
            + (f"<br><span style='opacity:.75'>{html.escape(i.advice)}</span>" if i.advice else "") + "</li>"
            for i in self.issues
        )
        return (f"<div style='font-family:system-ui,sans-serif;font-size:14px'>"
                f"<b style='color:{color}'>{icon} {html.escape(self.summary)}</b>"
                f"{'<ul>' + items + '</ul>' if items else ''}</div>")


def _stats(t: torch.Tensor) -> Dict[str, Any]:
    t = t.detach()
    if t.is_quantized:
        t = t.dequantize()
    if t.device.type == "meta":
        return {}
    f = t.float()
    finite = torch.isfinite(f)
    n_nan = int(torch.isnan(f).sum())
    n_inf = int(torch.isinf(f).sum())
    good = f[finite] if (n_nan or n_inf) else f.flatten()
    if good.numel() == 0:
        return {"nan": n_nan, "inf": n_inf, "abs_max": float("nan"), "std": float("nan"), "mean": float("nan"),
                "norm": float("nan"), "zero_frac": 0.0}
    return {
        "nan": n_nan,
        "inf": n_inf,
        "mean": float(good.mean()),
        "std": float(good.std()) if good.numel() > 1 else 0.0,
        "abs_max": float(good.abs().max()),
        "norm": float(good.norm()),
        "zero_frac": float((good == 0).float().mean()),
    }


def _weight_tensors(model: nn.Module):
    """(name, tensor, owning module, is_parameter) for parameters and quantized weights, tied weights once."""
    seen = set()
    for mod_name, m in model.named_modules():
        for p_name, p in m.named_parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            yield (f"{mod_name}.{p_name}" if mod_name else p_name), p, m, True
        if hasattr(m, "_packed_params") and callable(getattr(m, "weight", None)):
            try:
                yield f"{mod_name}.weight", m.weight(), m, False
            except Exception:
                pass


@torch.no_grad()
def health(model, *, extreme: float = 1e3, dead_row_frac: float = 0.05) -> HealthReport:
    """Check every weight (and gradient, if present) for common signs of a broken model.

    Args:
        model: Any ``nn.Module`` (or a :class:`llmreport.LanguageModel`).
        extreme: Weights larger than this in absolute value are reported.
        dead_row_frac: Report a matrix when more than this fraction of its rows are all zero.

    Returns:
        A :class:`HealthReport`. ``report.ok`` is True when nothing is wrong,
        ``report.issues`` lists problems, ``report.layers`` has per-tensor statistics.

    Example:
        >>> report = llmreport.health(model)
        >>> report.ok
        True
    """
    module = getattr(model, "module", model)
    if not isinstance(module, nn.Module):
        raise TypeError("health() needs a PyTorch nn.Module")
    issues: List[Issue] = []
    layers: List[Dict[str, Any]] = []
    nan_w, inf_w, zero_mats, collapsed, dead, big, fp16_overflow = [], [], [], [], [], [], []
    nan_g, zero_g, no_grad = [], [], []
    grad_sq, with_grad, trainable = 0.0, 0, 0
    abs_max, n_params = 0.0, 0
    dtypes, devices = set(), set()
    by_shape: Dict[tuple, List[tuple]] = {}

    for name, t, owner, is_param in _weight_tensors(module):
        st = _stats(t)
        dtypes.add(str(t.dtype).replace("torch.", ""))
        devices.add(str(t.device))
        n_params += t.numel()
        row = {"name": name, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), **st}
        if not st:
            layers.append(row)
            continue
        abs_max = max(abs_max, st["abs_max"]) if not math.isnan(st["abs_max"]) else abs_max
        if st["nan"]:
            nan_w.append(name)
        if st["inf"]:
            inf_w.append(name)
        is_matrix = t.dim() >= 2 and t.numel() > 1
        if is_matrix and st["abs_max"] == 0:
            zero_mats.append(name)
        if is_norm(owner) and name.endswith("weight") and t.dim() == 1 and st["abs_max"] < 1e-3:
            collapsed.append(name)
        if st["abs_max"] > extreme:
            big.append(name)
        if t.dtype in (torch.float16,) and st["abs_max"] > 6e4:
            fp16_overflow.append(name)
        if is_matrix and st["abs_max"] > 0:
            dense = t.detach().dequantize() if t.is_quantized else t.detach()
            rows = dense.float().reshape(t.shape[0], -1)
            zero_rows = (rows.abs().amax(dim=1) == 0)
            pad = getattr(owner, "padding_idx", None)
            if isinstance(owner, nn.Embedding) and pad is not None and 0 <= pad < zero_rows.numel():
                zero_rows[pad] = False  # the padding row is meant to be zero
            n_dead = int(zero_rows.sum())
            if n_dead > 1 and n_dead / zero_rows.numel() > dead_row_frac:
                dead.append(f"{name} ({n_dead}/{zero_rows.numel()} rows)")
            row["zero_rows"] = n_dead
            by_shape.setdefault(tuple(t.shape), []).append((name, st["std"]))
        if is_param and t.requires_grad:
            trainable += 1
            g = t.grad
            if g is None:
                no_grad.append(name)
            else:
                with_grad += 1
                gs = _stats(g)
                row["grad_norm"] = gs.get("norm")
                if gs.get("nan") or gs.get("inf"):
                    nan_g.append(name)
                elif gs.get("norm") == 0:
                    zero_g.append(name)
                else:
                    grad_sq += gs["norm"] ** 2
        layers.append(row)

    # Layers whose spread is wildly different from other layers of the same shape.
    outliers = []
    for shape, members in by_shape.items():
        if len(members) < 4:
            continue
        stds = [s for _, s in members if s > 0 and not math.isnan(s)]
        if len(stds) < 4:
            continue
        med = statistics.median(stds)
        for name, s in members:
            if med > 0 and (s > 50 * med or s < med / 50):
                outliers.append(name)

    if nan_w or inf_w:
        issues.append(Issue(CRITICAL, "Weights contain NaN or infinite values; the model's outputs are garbage.",
                            nan_w + inf_w, "Go back to the last good checkpoint and lower the learning rate or "
                            "enable gradient clipping."))
    if nan_g:
        issues.append(Issue(CRITICAL, "Gradients contain NaN or infinite values; the next optimizer step will "
                            "corrupt the weights.", nan_g, "Skip this step, lower the learning rate, check the "
                            "loss for overflow (use bfloat16 or loss scaling with float16)."))
    if fp16_overflow:
        issues.append(Issue(CRITICAL, "float16 weights are close to the largest float16 value (65,504) and will "
                            "overflow.", fp16_overflow, "Use bfloat16 or float32."))
    if zero_mats:
        issues.append(Issue(WARNING, "Weight matrices that are entirely zero (dead layers).", zero_mats,
                            "Check the initialisation and that these layers are being trained."))
    if collapsed:
        issues.append(Issue(WARNING, "Normalisation layers whose scale has collapsed to about zero; they block "
                            "the signal.", collapsed, "Usually caused by weight decay on norm weights; exclude "
                            "them from weight decay."))
    if dead:
        issues.append(Issue(WARNING, "Matrices with many all-zero rows (dead neurons or never-trained "
                            "embeddings).", dead))
    if big:
        issues.append(Issue(WARNING, f"Weights larger than {extreme:g} in absolute value.", big,
                            "Very large weights often come before a loss spike; consider gradient clipping or "
                            "weight decay."))
    if outliers:
        issues.append(Issue(WARNING, "Layers whose weight spread is over 50x different from other layers of the "
                            "same shape.", outliers))
    if zero_g:
        issues.append(Issue(WARNING, "Parameters whose gradient is exactly zero (vanishing gradients or unused "
                            "layers).", zero_g))
    if with_grad and no_grad:
        issues.append(Issue(INFO, "Trainable parameters that received no gradient in the last backward pass "
                            "(not used in the forward pass, or should be frozen).", no_grad))
    if len(dtypes) > 1:
        issues.append(Issue(INFO, f"Mixed dtypes in the weights: {', '.join(sorted(dtypes))}."))
    if len(devices) > 1:
        issues.append(Issue(INFO, f"Weights are spread over several devices: {', '.join(sorted(devices))}."))

    issues.sort(key=lambda i: _RANK[i.severity])
    status = "critical" if any(i.severity == CRITICAL for i in issues) else \
        "warning" if any(i.severity == WARNING for i in issues) else "ok"
    totals = {
        "tensors": len(layers), "params": n_params, "abs_max": abs_max, "trainable_tensors": trainable,
        "with_grad": with_grad, "grad_norm": math.sqrt(grad_sq) if with_grad else None,
        "dtypes": sorted(dtypes), "devices": sorted(devices),
    }
    return HealthReport(status=status, issues=issues, layers=layers, totals=totals)


def grad_norm(model) -> Optional[float]:
    """Total L2 norm of all gradients (None if no gradients exist). NaN if any gradient is NaN or infinite."""
    module = getattr(model, "module", model)
    total, found = 0.0, False
    for p in module.parameters():
        if p.grad is not None:
            found = True
            n = float(p.grad.detach().float().norm())
            if not math.isfinite(n):
                return float("nan")
            total += n * n
    return math.sqrt(total) if found else None


def weight_norm(model) -> float:
    """Total L2 norm of all weights."""
    module = getattr(model, "module", model)
    return math.sqrt(sum(float(p.detach().float().norm()) ** 2 for p in module.parameters()))
