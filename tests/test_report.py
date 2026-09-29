import json

import llmreport
from llmreport import Report
from llmreport.cli import main as cli_main


def test_default_report_runs_every_check(full_report):
    names = [r.name for r in full_report]
    assert names == llmreport.DEFAULT_CHECKS
    assert not [r for r in full_report if r.status == "error"], [r.summary for r in full_report]
    assert full_report.metadata["model"] == "tiny-random-gpt2"
    assert all(r.duration_s >= 0 for r in full_report)


def test_flat_metrics(full_report):
    m = full_report.metrics
    assert "architecture.total_params" in m
    assert "perplexity.perplexity" in m


def test_text_output(full_report):
    text = str(full_report)
    assert "tiny-random-gpt2" in text
    for r in full_report:
        assert r.title in text


def test_html_output(full_report, tmp_path):
    frag = full_report._repr_html_()
    assert frag.startswith("<style>") and "llmr-score" in frag
    path = tmp_path / "r.html"
    page = full_report.to_html(str(path))
    assert path.read_text(encoding="utf-8") == page
    assert page.startswith("<!DOCTYPE html>") and "</html>" in page
    # User-controlled text (model outputs) must be escaped.
    assert "<script" not in page.lower()


def test_html_escapes_untrusted_text():
    r = llmreport.AnalyzerResult(name="x", title="X", summary="<script>alert(1)</script>",
                                 details={"samples": [{"output": "<img src=x onerror=alert(1)>"}]})
    page = Report([r], {"model": "<b>m</b>"}).to_html()
    assert "<script>alert" not in page and "<img src=x" not in page and "<b>m</b>" not in page


def test_markdown_model_card(full_report, tmp_path):
    md = full_report.to_markdown(str(tmp_path / "card.md"))
    assert md.startswith("---\nlibrary_name: transformers")
    assert "## Intended use" in md and "GPT2LMHeadModel" in md
    plain = full_report.to_markdown(model_card=False)
    assert plain.startswith("# llmreport:")


def test_json_roundtrip(full_report, tmp_path):
    path = tmp_path / "r.json"
    full_report.to_json(str(path))
    data = json.loads(path.read_text())
    assert data["metadata"]["model"] == "tiny-random-gpt2"
    loaded = Report.load(str(path))
    assert loaded.metrics == json.loads(json.dumps(full_report.metrics))
    assert [r.name for r in loaded] == [r.name for r in full_report]


def test_compare(full_report):
    before = Report.from_dict(json.loads(full_report.to_json()))
    after = Report.from_dict(json.loads(full_report.to_json()))
    after["perplexity"].metrics["perplexity"] = before["perplexity"].metrics["perplexity"] / 2
    comp = before.compare(after, names=("before", "after"))
    row = next(r for r in comp.rows if r["metric"] == "perplexity.perplexity")
    assert row["verdict"] == "better"
    assert "before" in str(comp) and "<table>" in comp._repr_html_()


def test_show_does_not_crash(full_report, capsys):
    full_report.show()
    assert "tiny-random-gpt2" in capsys.readouterr().out


def test_cli_list_checks(capsys):
    assert cli_main(["--list-checks"]) == 0
    out = capsys.readouterr().out
    assert "perplexity" in out and "refusal" in out


def test_compare_ignores_noise(full_report):
    a = Report.from_dict(json.loads(full_report.to_json()))
    b = Report.from_dict(json.loads(full_report.to_json()))
    b["performance"].metrics["ttft_ms"] = a["performance"].metrics["ttft_ms"] * 1.05  # 5% timing jitter
    b["repetition"].metrics["repeated_3gram_rate"] = a["repetition"].metrics["repeated_3gram_rate"] + 0.004
    rows = {r["metric"]: r["verdict"] for r in a.compare(b).rows}
    assert rows["performance.ttft_ms"] == "" and rows["repetition.repeated_3gram_rate"] == ""
