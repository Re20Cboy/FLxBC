from pathlib import Path


def test_dashboard_quality_chart_emphasizes_primary_metrics_and_hides_auc():
    source = Path("dashboard/app.py").read_text()

    assert "import plotly.graph_objects as go" in source
    assert "LINE_STYLES" in source
    assert '"test_accuracy": {"color": "#1f5fbf", "width": 4' in source
    assert '"test_macro_f1": {"color": "#d33f32", "width": 4' in source

    quality_start = source.index('title="质量指标"')
    quality_end = source.index('empty_message="该 run 尚未记录 accuracy / macro-F1 指标。"')
    quality_block = source[quality_start:quality_end]
    assert '"test_auc"' not in quality_block
