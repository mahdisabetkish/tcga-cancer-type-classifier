"""Assemble dashboard/src/* + dashboard/data/model_bundle.json into a single
self-contained dashboard/index.html (no external requests, no build tooling)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "dashboard" / "src"
DATA = ROOT / "dashboard" / "data" / "model_bundle.json"
OUT = ROOT / "dashboard" / "index.html"

template = (SRC / "index.template.html").read_text()
styles = (SRC / "styles.css").read_text()
charts_js = (SRC / "charts.js").read_text()
app_js = (SRC / "app.js").read_text()
data_json = DATA.read_text()

html = template.replace("{{STYLES}}", styles)
html = html.replace("{{DATA_JSON}}", data_json)
html = html.replace("{{CHARTS_JS}}", charts_js)
html = html.replace("{{APP_JS}}", app_js)

OUT.write_text(html)
size_mb = OUT.stat().st_size / 1e6
print(f"Wrote {OUT} ({size_mb:.2f} MB)")
