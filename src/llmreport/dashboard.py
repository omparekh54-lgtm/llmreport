"""The training dashboard: one self-contained HTML page (no internet needed) with live charts."""

from __future__ import annotations

import html
import json
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CSS = """
.lrd{--surface:#fcfcfb;--panel:#ffffff;--border:#e4e3df;--grid:#ecebe7;--text:#0b0b0b;--text2:#52514e;
--muted:#8a8984;--series1:#2a78d6;--raw:#b9b8b2;--critical:#c62828;--warning:#9a6200;--info:#2a6bb8;--ok:#1a7f37;
color-scheme:light;background:var(--surface);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,
Roboto,sans-serif;padding:20px 16px;min-height:100vh;box-sizing:border-box}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .lrd{--surface:#1a1a19;--panel:#222221;
--border:#3a3a38;--grid:#2e2e2c;--text:#ffffff;--text2:#c3c2b7;--muted:#8f8e86;--series1:#3987e5;--raw:#5f5e59;
--critical:#ff6b6b;--warning:#e0a526;--info:#6ea8f0;--ok:#4cc26a;color-scheme:dark}}
:root[data-theme="dark"] .lrd{--surface:#1a1a19;--panel:#222221;--border:#3a3a38;--grid:#2e2e2c;--text:#ffffff;
--text2:#c3c2b7;--muted:#8f8e86;--series1:#3987e5;--raw:#5f5e59;--critical:#ff6b6b;--warning:#e0a526;
--info:#6ea8f0;--ok:#4cc26a;color-scheme:dark}
.lrd *{box-sizing:border-box}.lrd h1{font-size:20px;margin:0}.lrd h2{font-size:15px;margin:24px 0 10px}
.lrd .sub{color:var(--text2);font-size:13px;margin-top:2px}.lrd .pill{display:inline-block;padding:1px 8px;
border-radius:999px;font-size:12px;border:1px solid var(--border);color:var(--text2);margin-left:8px;
vertical-align:2px}.lrd .pill.live{color:var(--ok);border-color:var(--ok)}
.lrd .kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:16px}
.lrd .kpi{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.lrd .kpi .l{color:var(--text2);font-size:12px}.lrd .kpi .v{font-size:20px;font-weight:600;margin-top:2px;
font-variant-numeric:tabular-nums}.lrd .kpi .s{color:var(--muted);font-size:12px}
.lrd .bar{height:4px;background:var(--grid);border-radius:2px;margin-top:6px}.lrd .bar i{display:block;height:4px;
background:var(--series1);border-radius:2px}
.lrd .banner{margin-top:16px;padding:10px 12px;border-radius:10px;border:1px solid var(--critical);
color:var(--critical);background:color-mix(in srgb,var(--critical) 8%,transparent)}
.lrd .charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.lrd .chart{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 12px;
position:relative}.lrd .chart .t{font-weight:600;font-size:13px}.lrd .chart .d{color:var(--text2);font-size:12px}
.lrd .legend{display:flex;gap:12px;font-size:12px;color:var(--text2);margin-top:2px}.lrd .legend span{display:
inline-flex;align-items:center;gap:5px}.lrd .legend i{display:inline-block;width:14px;height:2px;border-radius:1px}
.lrd svg{display:block;width:100%;height:auto;overflow:visible}.lrd svg text{fill:var(--muted);font-size:10px}
.lrd .tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--border);
border-radius:6px;padding:4px 8px;font-size:12px;white-space:nowrap;display:none;color:var(--text);
box-shadow:0 2px 8px rgba(0,0,0,.12);font-variant-numeric:tabular-nums}
.lrd table{width:100%;border-collapse:collapse;font-size:13px}.lrd td,.lrd th{text-align:left;padding:6px 8px;
border-bottom:1px solid var(--grid);vertical-align:top}.lrd th{color:var(--text2);font-weight:500}
.lrd .sev{font-weight:600;white-space:nowrap}.lrd .sev.critical{color:var(--critical)}.lrd .sev.warning{
color:var(--warning)}.lrd .sev.info{color:var(--info)}
.lrd pre{white-space:pre-wrap;margin:4px 0 0;background:var(--surface);border:1px solid var(--grid);
border-radius:6px;padding:8px;font-size:12px;max-height:220px;overflow:auto}
.lrd .box{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:4px 8px;overflow-x:auto}
.lrd .empty{color:var(--muted);padding:8px}.lrd .foot{color:var(--muted);font-size:12px;margin-top:24px}
"""

_JS = """
document.querySelectorAll('.lrd .chart').forEach(function(c){
  var svg=c.querySelector('svg'), tip=c.querySelector('.tip'); if(!svg) return;
  var pts=JSON.parse(svg.getAttribute('data-points')||'[]'), cross=svg.querySelector('.cross');
  var dot=svg.querySelector('.hover-dot'); if(!pts.length) return;
  function hide(){tip.style.display='none';cross.setAttribute('opacity',0);dot.setAttribute('opacity',0);}
  svg.addEventListener('mousemove',function(e){
    var r=svg.getBoundingClientRect(), vb=svg.viewBox.baseVal, x=(e.clientX-r.left)*vb.width/r.width;
    var best=pts[0]; for(var i=1;i<pts.length;i++){if(Math.abs(pts[i][0]-x)<Math.abs(best[0]-x))best=pts[i];}
    cross.setAttribute('x1',best[0]);cross.setAttribute('x2',best[0]);cross.setAttribute('opacity',1);
    dot.setAttribute('cx',best[0]);dot.setAttribute('cy',best[1]);dot.setAttribute('opacity',1);
    tip.textContent=best[2]; tip.style.display='block';
    var cr=c.getBoundingClientRect();
    var left=r.left-cr.left+best[0]*r.width/vb.width, top=r.top-cr.top+best[1]*r.height/vb.height;
    tip.style.left=Math.max(4,Math.min(left+10,c.clientWidth-tip.offsetWidth-4))+'px';
    tip.style.top=Math.max(4,top-tip.offsetHeight-8)+'px';
  });
  svg.addEventListener('mouseleave',hide);
});
"""

_W, _H, _PAD_L, _PAD_R, _PAD_T, _PAD_B = 320.0, 150.0, 44.0, 8.0, 8.0, 20.0


def _esc(x) -> str:
    return html.escape(str(x))


def _num(v: Optional[float], digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e6):
        return f"{v:.2e}"
    if a >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}g}"


def _bytes(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(v) < 1024 or unit == "TB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024
    return f"{v:.1f} TB"


def _duration(s: Optional[float]) -> str:
    if s is None:
        return "n/a"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {sec:02d}s" if m else f"{sec}s"


def decimate(points: List[Tuple[float, float]], limit: int = 600) -> List[Tuple[float, float]]:
    """Reduce to about ``limit`` points while keeping every spike (min and max of each bucket)."""
    points = [(x, y) for x, y in points if y is not None and math.isfinite(y)]
    if len(points) <= limit:
        return points
    size = math.ceil(len(points) / (limit / 2))
    out = []
    for i in range(0, len(points), size):
        bucket = points[i:i + size]
        lo = min(bucket, key=lambda p: p[1])
        hi = max(bucket, key=lambda p: p[1])
        out.extend(sorted({lo, hi}, key=lambda p: p[0]))
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def _nice_ticks(lo: float, hi: float, n: int = 4) -> List[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    ticks = []
    t = first
    while t <= hi + step * 1e-9 and len(ticks) < 10:
        ticks.append(round(t, 12))
        t += step
    return ticks


def line_chart(title: str, series: Sequence[Dict[str, Any]], *, description: str = "", fmt=_num,
               log: Optional[bool] = None) -> str:
    """One chart: one y-axis, one or more series [{label, points, color, width, main}]."""
    all_pts = [p for s in series for p in s["points"]]
    if not all_pts:
        return ""
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x1 = x0 + 1
    positive = all(y > 0 for y in ys)
    if log is None:
        log = positive and max(ys) / max(min(ys), 1e-300) > 200
    tf = (lambda v: math.log10(v)) if log else (lambda v: v)
    ty = [tf(y) for y in ys]
    y0, y1 = min(ty), max(ty)
    if y1 == y0:
        y0, y1 = y0 - (abs(y0) * 0.1 or 1), y1 + (abs(y1) * 0.1 or 1)
    pad = (y1 - y0) * 0.06
    y0, y1 = y0 - pad, y1 + pad
    if not log and min(ys) >= 0 and y0 < 0:
        y0 = 0.0
    pw, ph = _W - _PAD_L - _PAD_R, _H - _PAD_T - _PAD_B

    def px(x):
        return _PAD_L + (x - x0) / (x1 - x0) * pw

    def py(y):
        return _PAD_T + (1 - (tf(y) - y0) / (y1 - y0)) * ph

    grid = ""
    if log:
        ticks = [10 ** e for e in range(math.floor(y0), math.ceil(y1) + 1)]
        ticks = [t for t in ticks if y0 <= math.log10(t) <= y1] or [10 ** y0, 10 ** y1]
    else:
        ticks = _nice_ticks(y0, y1)
    tick_fmt = (lambda v: f"{v:g}" if 1e-2 <= v <= 1e4 else f"{v:.0e}") if log else fmt
    for t in ticks:
        y = py(t)
        grid += (f"<line x1='{_PAD_L}' x2='{_W - _PAD_R}' y1='{y:.1f}' y2='{y:.1f}' stroke='var(--grid)' "
                 f"stroke-width='1'/><text x='{_PAD_L - 6}' y='{y + 3:.1f}' text-anchor='end'>{_esc(tick_fmt(t))}</text>")
    for t in _nice_ticks(x0, x1, 4):
        x = px(t)
        grid += f"<text x='{x:.1f}' y='{_H - 4}' text-anchor='middle'>{_esc(_num(t))}</text>"
    paths, hover = "", []
    for s in series:
        pts = s["points"]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(pts))
        paths += (f"<path d='{d}' fill='none' stroke='{s['color']}' stroke-width='{s.get('width', 2)}' "
                  "stroke-linejoin='round' stroke-linecap='round'/>")
        if s.get("dots"):
            paths += "".join(f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='3.5' fill='{s['color']}' "
                             "stroke='var(--panel)' stroke-width='2'/>" for x, y in pts)
        if s.get("main", True):
            hover = [[round(px(x), 1), round(py(y), 1), f"step {int(x):,} · {s['label']} {fmt(y)}"] for x, y in pts]
    legend = ""
    if len(series) > 1:
        legend = "<div class='legend'>" + "".join(
            f"<span><i style='background:{s['color']}'></i>{_esc(s['label'])}</span>" for s in series) + "</div>"
    last = next((s for s in series if s.get("main", True)), series[0])
    latest = fmt(last["points"][-1][1]) if last["points"] else ""
    svg = (f"<svg viewBox='0 0 {_W:.0f} {_H:.0f}' role='img' aria-label='{_esc(title)}' "
           f"data-points='{_esc(json.dumps(hover))}'>{grid}{paths}"
           f"<line class='cross' y1='{_PAD_T}' y2='{_H - _PAD_B}' stroke='var(--muted)' stroke-width='1' opacity='0'/>"
           f"<circle class='hover-dot' r='4' fill='var(--series1)' stroke='var(--panel)' stroke-width='2' "
           f"opacity='0'/></svg>")
    return (f"<div class='chart'><div class='t'>{_esc(title)} <span style='float:right;font-weight:400;"
            f"color:var(--text2)'>{_esc(latest)}</span></div>"
            + (f"<div class='d'>{_esc(description)}{' (log scale)' if log else ''}</div>" if description or log else "")
            + f"{legend}{svg}<div class='tip'></div></div>")


def _series(rows, key) -> List[Tuple[float, float]]:
    return [(r["step"], r[key]) for r in rows if isinstance(r.get(key), (int, float)) and r.get("step") is not None]


_STEP_KEYS_DONE = {"step", "time", "loss", "loss_avg", "lr", "grad_norm", "tokens_per_s", "samples_per_s",
                   "step_time", "tokens", "gpu_mem_bytes", "gpu_peak_bytes"}
_EVAL_TITLES = {"eval_perplexity": ("Evaluation perplexity", "On your evaluation texts; lower is better"),
                "code_pass_rate": ("Code pass rate", "Share of small Python tasks passing their tests"),
                "eval_bits_per_char": ("Evaluation bits per character", "Tokenizer-independent; lower is better"),
                "sample_repetition": ("Repetition in samples", "Share of repeated 3-word phrases; lower is better")}


def _charts(rows, events, compact=False) -> str:
    out = []
    raw = decimate(_series(rows, "loss"))
    avg = decimate(_series(rows, "loss_avg"))
    if raw:
        series = [{"label": "loss", "points": raw, "color": "var(--raw)", "width": 1, "main": not avg}]
        if avg:
            series.append({"label": "average", "points": avg, "color": "var(--series1)", "width": 2})
        out.append(line_chart("Training loss", series, description="Per step, with a smoothed average"))
    if compact:
        return "".join(out)
    evals = [e for e in events if e.get("kind") == "eval"]
    eval_keys = []
    for e in evals:
        for k, v in e.items():
            if k not in ("kind", "step", "time", "eval_seconds", "eval_loss") and isinstance(v, (int, float)) \
                    and k not in eval_keys:
                eval_keys.append(k)
    for k in eval_keys:
        pts = _series(evals, k)
        title, desc = _EVAL_TITLES.get(k, (k.replace("_", " ").capitalize(), "Logged evaluation"))
        fmt = (lambda v: f"{v:.0%}") if k in ("code_pass_rate", "sample_repetition") else _num
        out.append(line_chart(title, [{"label": k.replace("_", " "), "points": pts, "color": "var(--series1)",
                                       "dots": len(pts) <= 60}], description=desc, fmt=fmt))
    for key, title, desc, fmt in (
        ("lr", "Learning rate", "From the optimizer", _num),
        ("grad_norm", "Gradient norm", "Total L2 norm of all gradients", _num),
        ("tokens_per_s", "Speed", "Tokens per second", _num),
        ("step_time", "Step time", "Seconds per step", _num),
        ("gpu_mem_bytes", "GPU memory", "Allocated by PyTorch", _bytes),
    ):
        pts = decimate(_series(rows, key))
        if key == "step_time" and _series(rows, "tokens_per_s"):
            continue
        if pts:
            out.append(line_chart(title, [{"label": title.lower(), "points": pts, "color": "var(--series1)"}],
                                  description=desc, fmt=fmt))
    extra = []
    for r in rows[-200:]:
        extra += [k for k, v in r.items() if k not in _STEP_KEYS_DONE and isinstance(v, (int, float))
                  and k not in extra]
    for k in extra:
        pts = decimate(_series(rows, k))
        if pts:
            out.append(line_chart(k.replace("_", " ").capitalize(), [{"label": k, "points": pts,
                                                                       "color": "var(--series1)"}]))
    return "".join(out)


def _kpis(meta, rows, events) -> str:
    from .tracking import run_summary

    s = run_summary(rows, events, meta.get("total_steps"))
    tiles = []
    step_sub, bar = "", ""
    if s.get("total_steps"):
        step_sub = f"of {s['total_steps']:,} ({s['progress']:.0%})"
        bar = f"<div class='bar'><i style='width:{min(100, s['progress'] * 100):.1f}%'></i></div>"
    tiles.append(("Step", f"{s['steps']:,}", step_sub, bar))
    if "last_loss_avg" in s:
        tiles.append(("Loss (average)", _num(s["last_loss_avg"]), f"first {_num(s.get('first_loss'))}", ""))
    if "best_loss" in s:
        tiles.append(("Best loss", _num(s["best_loss"]), f"at step {s['best_loss_step']:,}", ""))
    if "best_eval_perplexity" in s:
        last = s.get("last_eval", {}).get("eval_perplexity")
        tiles.append(("Eval perplexity", _num(last), f"best {_num(s['best_eval_perplexity'])} at step "
                      f"{s['best_eval_step']:,}", ""))
    if "tokens_per_s" in s:
        tiles.append(("Speed", f"{s['tokens_per_s']:,.0f}", "tokens / second", ""))
    if "tokens_seen" in s:
        tiles.append(("Tokens seen", _num(float(s["tokens_seen"]), 3), "", ""))
    if "elapsed_s" in s:
        eta = ""
        times = [r["step_time"] for r in rows[-50:] if r.get("step_time")]
        if s.get("total_steps") and times:
            times.sort()
            eta = f"about {_duration(max(0, s['total_steps'] - s['steps']) * times[len(times) // 2])} left"
        tiles.append(("Elapsed", _duration(s["elapsed_s"]), eta, ""))
    a = s["alerts"]
    tiles.append(("Alerts", str(sum(a.values())), f"{a['critical']} critical · {a['warning']} warning", ""))
    return "<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='l'>{_esc(label)}</div><div class='v'>{_esc(v)}</div>"
        f"<div class='s'>{_esc(sub)}</div>{bar}</div>" for label, v, sub, bar in tiles) + "</div>"


def _alerts_table(events, limit=30) -> str:
    alerts = [e for e in events if e.get("kind") == "alert"][::-1][:limit]
    if not alerts:
        return "<div class='box'><div class='empty'>✓ No problems detected.</div></div>"
    icon = {"critical": "✕", "warning": "!", "info": "i"}
    rows = "".join(
        f"<tr><td class='sev {_esc(a.get('severity'))}'>{icon.get(a.get('severity'), 'i')} "
        f"{_esc(str(a.get('severity', '')).capitalize())}</td><td>{a.get('step', 0):,}</td>"
        f"<td>{_esc(a.get('message', ''))}</td></tr>" for a in alerts)
    return f"<div class='box'><table><tr><th>Severity</th><th>Step</th><th>What happened</th></tr>{rows}</table></div>"


def _samples(events) -> str:
    latest = next((e for e in reversed(events) if e.get("kind") == "samples"), None)
    if not latest:
        return ""
    items = "".join(f"<div style='margin:8px 0'><b>{_esc(s.get('prompt', ''))}</b><pre>{_esc(s.get('output', ''))}"
                    f"</pre></div>" for s in latest.get("samples", []))
    return f"<h2>Latest samples <span class='pill'>step {latest.get('step', 0):,}</span></h2><div class='box'>{items}</div>"


def _model_box(meta) -> str:
    m = meta.get("model") or {}
    if not m:
        return ""
    fields = [("Model", m.get("model_class")), ("Kind", m.get("kind")), ("Family", m.get("family")),
              ("Parameters", f"{m['total_params']:,}" if isinstance(m.get("total_params"), int) else None),
              ("Layers", m.get("layers")), ("Hidden size", m.get("hidden_size")),
              ("Attention", m.get("attention_type")), ("Context", m.get("max_context")),
              ("Device", m.get("device")), ("Weights", _bytes(m["weights_bytes"]) if m.get("weights_bytes") else None),
              ("Health at start", m.get("health"))]
    rows = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in fields if v not in (None, ""))
    return f"<h2>Model</h2><div class='box'><table>{rows}</table></div>"


def render_dashboard(meta: Dict[str, Any], rows: List[Dict[str, Any]], events: List[Dict[str, Any]],
                     live: bool = True, refresh_seconds: int = 15) -> str:
    """The full dashboard page."""
    name = meta.get("name") or "training run"
    updated = time.strftime("%H:%M:%S")
    status = "<span class='pill live'>● live</span>" if live else "<span class='pill'>finished</span>"
    critical = [e for e in events if e.get("kind") == "alert" and e.get("severity") == "critical"]
    banner = (f"<div class='banner'>✕ Critical at step {critical[-1].get('step', 0):,}: "
              f"{_esc(critical[-1].get('message', ''))}</div>") if critical else ""
    health = next((e for e in reversed(events) if e.get("kind") == "health"), None)
    health_html = ""
    if health:
        color = {"ok": "var(--ok)", "warning": "var(--warning)", "critical": "var(--critical)"}.get(
            health.get("status"), "var(--text2)")
        issues = "".join(f"<li>{_esc(i)}</li>" for i in health.get("issues", []))
        health_html = (f"<h2>Weight health <span class='pill'>step {health.get('step', 0):,}</span></h2>"
                       f"<div class='box' style='padding:10px 12px'><b style='color:{color}'>"
                       f"{_esc(health.get('summary', ''))}</b>{'<ul>' + issues + '</ul>' if issues else ''}</div>")
    charts = _charts(rows, events)
    body = (f"<div class='lrd'><h1>{_esc(name)}{status}</h1>"
            f"<div class='sub'>llmreport training dashboard · updated {updated}"
            f"{f' · refreshes every {refresh_seconds}s' if live else ''}</div>{banner}{_kpis(meta, rows, events)}"
            f"<h2>Charts</h2><div class='charts'>{charts or '<div class=empty>No data yet.</div>'}</div>"
            f"<h2>Alerts</h2>{_alerts_table(events)}{_samples(events)}{health_html}{_model_box(meta)}"
            f"<div class='foot'>Hover a chart for exact values. Data: metrics.jsonl and events.jsonl in this "
            f"folder.</div></div>")
    refresh = f"<meta http-equiv='refresh' content='{refresh_seconds}'>" if live else ""
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"{refresh}<title>{_esc(name)} · training</title><style>body{{margin:0}}{_CSS}</style></head>"
            f"<body>{body}<script>{_JS}</script></body></html>")


def render_panel(tracker) -> str:
    """A compact live panel for Jupyter notebooks."""
    critical = [e for e in tracker.events if e.get("kind") == "alert"][-3:]
    alerts = "".join(f"<div class='sev {_esc(a.get('severity'))}'>step {a.get('step', 0):,}: "
                     f"{_esc(a.get('message', ''))}</div>" for a in critical)
    return (f"<style>{_CSS}</style><div class='lrd' style='min-height:0;padding:12px'>"
            f"<b>{_esc(tracker.name)}</b>{_kpis(tracker.meta, tracker.rows, tracker.events)}"
            f"<div class='charts' style='margin-top:10px'>{_charts(tracker.rows, tracker.events, compact=True)}</div>"
            f"{alerts}<div class='foot' style='margin-top:6px'>Full dashboard: {_esc(tracker.dashboard_path)}</div>"
            f"</div>")
