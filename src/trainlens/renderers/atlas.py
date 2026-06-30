"""Radio-map inspired notebook HTML renderer."""

from __future__ import annotations

from hashlib import sha256
from html import escape

from trainlens.models.analysis import AnalysisResult


class AtlasRenderer:
    """Render analysis results as a notebook-local training atlas."""

    def render(self, result: AnalysisResult) -> str:
        model = escape(result.model_name or "Notebook training run")
        framework = escape(result.framework or "detected from notebook context")
        status = escape(_status_label(result))
        points = "\n".join(
            _point_html(label, index) for index, label in enumerate(_point_labels(result))
        )
        metrics = "\n".join(
            f'<li><span>{escape(name)}</span><strong>{value:.3f}</strong></li>'
            for name, value in sorted(result.metrics.items())[:4]
        )
        if not metrics:
            metrics = "<li><span>metrics</span><strong>pending</strong></li>"
        next_step = escape(
            result.recommendations[0].action
            if result.recommendations
            else "Expose metrics, dataset notes, and trace logs in the notebook."
        )
        signals = len(result.signals)
        trace_events = len(result.trace)
        return f"""<div class="trainlens-atlas" role="region" aria-label="TrainLens training atlas">
  <style>
    .trainlens-atlas {{
      position: relative;
      min-height: 430px;
      overflow: hidden;
      border-radius: 18px;
      color: #f8fafc;
      background:
        radial-gradient(circle at 28% 24%, rgba(72, 94, 63, .88), transparent 18%),
        radial-gradient(circle at 69% 31%, rgba(62, 88, 58, .9), transparent 20%),
        radial-gradient(circle at 46% 66%, rgba(168, 138, 86, .72), transparent 24%),
        radial-gradient(circle at 61% 72%, rgba(8, 20, 34, .92), transparent 17%),
        linear-gradient(135deg, #08101d 0%, #122018 42%, #263016 62%, #101827 100%);
      box-shadow: inset 0 0 80px rgba(0, 0, 0, .55), 0 20px 60px rgba(15, 23, 42, .25);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }}
    .trainlens-atlas::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(24deg, transparent 0 32%, rgba(9, 16, 30, .72) 33% 37%, transparent 38%),
        linear-gradient(112deg, transparent 0 48%, rgba(3, 9, 18, .6) 49% 53%, transparent 54%);
      opacity: .85;
    }}
    .trainlens-atlas__point {{
      position: absolute;
      left: var(--x);
      top: var(--y);
      width: var(--s);
      height: var(--s);
      border-radius: 999px;
      background: #6eff8f;
      box-shadow: 0 0 8px #6eff8f, 0 0 18px rgba(110, 255, 143, .55);
      transform: translate(-50%, -50%);
    }}
    .trainlens-atlas__focus {{
      position: absolute;
      left: 62%;
      top: 47%;
      width: 78px;
      height: 78px;
      border: 2px solid rgba(255, 255, 255, .9);
      border-radius: 999px;
      box-shadow: 0 0 0 1px rgba(15, 23, 42, .4), 0 0 24px rgba(255, 255, 255, .22);
      transform: translate(-50%, -50%);
    }}
    .trainlens-atlas__hud {{
      position: absolute;
      left: 16px;
      right: 16px;
      bottom: 16px;
      display: grid;
      grid-template-columns: minmax(220px, 330px) minmax(220px, 1fr);
      gap: 12px;
      align-items: end;
    }}
    .trainlens-atlas__card {{
      min-width: 0;
      border-radius: 10px;
      background: rgba(9, 13, 20, .84);
      border: 1px solid rgba(148, 163, 184, .16);
      box-shadow: 0 12px 32px rgba(0, 0, 0, .28);
      backdrop-filter: blur(12px);
      padding: 14px;
    }}
    .trainlens-atlas__eyebrow {{
      color: #6eff8f;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    .trainlens-atlas h3 {{
      margin: 5px 0 2px;
      font-size: 22px;
      line-height: 1.1;
    }}
    .trainlens-atlas p {{
      margin: 0;
      color: #dbeafe;
      font-size: 13px;
    }}
    .trainlens-atlas__stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .trainlens-atlas__stat {{
      border-radius: 8px;
      background: rgba(15, 23, 42, .72);
      padding: 8px;
    }}
    .trainlens-atlas__stat strong {{
      display: block;
      font-size: 18px;
    }}
    .trainlens-atlas__stat span {{
      color: #9ca3af;
      font-size: 11px;
    }}
    .trainlens-atlas__metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      list-style: none;
      margin: 10px 0 0;
      padding: 0;
    }}
    .trainlens-atlas__metrics li {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      border-radius: 8px;
      background: rgba(15, 23, 42, .72);
      padding: 8px 10px;
      font-size: 12px;
    }}
    .trainlens-atlas__metrics span {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #cbd5e1;
    }}
    .trainlens-atlas__metrics strong {{
      color: #6eff8f;
    }}
    @media (max-width: 720px) {{
      .trainlens-atlas {{
        min-height: 560px;
      }}
      .trainlens-atlas__hud {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
  {points}
  <div class="trainlens-atlas__focus" aria-hidden="true"></div>
  <div class="trainlens-atlas__hud">
    <section class="trainlens-atlas__card">
      <div class="trainlens-atlas__eyebrow">TrainLens atlas</div>
      <h3>{model}</h3>
      <p>{framework}</p>
      <div class="trainlens-atlas__stats">
        <div class="trainlens-atlas__stat">
          <strong>{len(result.metrics)}</strong><span>metrics</span>
        </div>
        <div class="trainlens-atlas__stat">
          <strong>{trace_events}</strong><span>trace events</span>
        </div>
        <div class="trainlens-atlas__stat"><strong>{signals}</strong><span>signals</span></div>
      </div>
    </section>
    <section class="trainlens-atlas__card">
      <div class="trainlens-atlas__eyebrow">{status}</div>
      <p>{next_step}</p>
      <ul class="trainlens-atlas__metrics">{metrics}</ul>
    </section>
  </div>
</div>"""


def _point_labels(result: AnalysisResult) -> list[str]:
    labels = list(result.metrics)
    labels.extend(signal.title for signal in result.signals)
    labels.extend(
        event.name or event.message or f"trace-{index}"
        for index, event in enumerate(result.trace)
    )
    labels.extend(result.top_features)
    if not labels:
        labels = ["notebook", "dataset", "metrics", "model", "trace", "recommendation"]
    return labels[:80]


def _point_html(label: str, index: int) -> str:
    digest = sha256(f"{index}:{label}".encode()).digest()
    x = 7 + digest[0] / 255 * 86
    y = 8 + digest[1] / 255 * 72
    size = 3 + digest[2] % 5
    return (
        '<span class="trainlens-atlas__point" '
        f'style="--x:{x:.2f}%;--y:{y:.2f}%;--s:{size}px" '
        f'title="{escape(label)}"></span>'
    )


def _status_label(result: AnalysisResult) -> str:
    if any(signal.severity == "critical" for signal in result.signals):
        return "Critical training signal"
    if any(signal.severity == "warning" for signal in result.signals):
        return "Training signal detected"
    if result.recommendations:
        return "Next experiment"
    return "Notebook context"
