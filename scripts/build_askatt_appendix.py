#!/usr/bin/env python3
"""Build the standalone AskATT System1 API appendix PDF."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "appendix-askatt-system1-api.pdf"

NAVY = colors.HexColor("#172554")
BLUE = colors.HexColor("#075985")
TEAL = colors.HexColor("#0F766E")
PURPLE = colors.HexColor("#7C3AED")
ORANGE = colors.HexColor("#C76F00")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#64748B")
PALE = colors.HexColor("#F1F5F9")
PALE_BLUE = colors.HexColor("#E0F2FE")
GREEN = colors.HexColor("#15803D")


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bolds = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    monos = [
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]
    normal = next((p for p in candidates if p.exists()), None)
    bold = next((p for p in bolds if p.exists()), None)
    mono = next((p for p in monos if p.exists()), None)
    if normal and bold and mono:
        pdfmetrics.registerFont(TTFont("AskSans", str(normal)))
        pdfmetrics.registerFont(TTFont("AskSansBold", str(bold)))
        pdfmetrics.registerFont(TTFont("AskMono", str(mono)))
        return "AskSans", "AskSansBold", "AskMono"
    return "Helvetica", "Helvetica-Bold", "Courier"


FONT, BOLD, MONO = register_fonts()


class Architecture(Flowable):
    def __init__(self, width: float = 500, height: float = 220) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        nodes = [
            (4, 158, 78, 42, "JEV client", BLUE),
            (99, 158, 90, 42, "POST\n/systemone", NAVY),
            (207, 158, 92, 42, "Validate +\ncompile", TEAL),
            (350, 158, 91, 42, "GLiClass\nshared pass", PURPLE),
            (460, 158, 40, 42, "Answers", GREEN),
        ]
        for x, y, w, h, label, fill in nodes:
            c.setFillColor(fill)
            c.roundRect(x, y, w, h, 6, stroke=0, fill=1)
            c.setFillColor(colors.white)
            c.setFont(BOLD, 8.5)
            lines = label.split("\n")
            for index, line in enumerate(lines):
                c.drawCentredString(x + w / 2, y + h / 2 + 5 - index * 11, line)
        c.setStrokeColor(MUTED)
        c.setLineWidth(1.5)
        for start, end in ((82, 99), (189, 207), (299, 350), (441, 460)):
            c.line(start, 179, end - 4, 179)
            c.line(end - 8, 183, end - 4, 179)
            c.line(end - 8, 175, end - 4, 179)

        branches = [
            (104, "Noul", "positive label", TEAL),
            (56, "Binary Choice", "p and 1-p", BLUE),
            (8, "General Choice", "one label/option", PURPLE),
        ]
        for y, title, detail, fill in branches:
            c.setFillColor(PALE)
            c.setStrokeColor(fill)
            c.roundRect(205, y, 140, 38, 5, stroke=1, fill=1)
            c.setFillColor(fill)
            c.setFont(BOLD, 8)
            c.drawString(214, y + 23, title)
            c.setFillColor(INK)
            c.setFont(FONT, 7.5)
            c.drawString(214, y + 10, detail)
            c.setStrokeColor(MUTED)
            c.line(253, 158, 253, y + 38)
            c.line(345, y + 19, 395, 158)
        c.setFillColor(ORANGE)
        c.setFont(BOLD, 8)
        c.drawString(350, 137, "State repeats only when labels exceed a pass")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName=BOLD, fontSize=27, leading=31, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName=FONT, fontSize=12, leading=17, textColor=MUTED, spaceAfter=14),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName=BOLD, fontSize=18, leading=22, textColor=NAVY, spaceBefore=7, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName=BOLD, fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=7, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName=FONT, fontSize=9.2, leading=13.5, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName=FONT, fontSize=7.8, leading=10.5, textColor=MUTED, spaceAfter=5),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName=BOLD, fontSize=11, leading=15, textColor=NAVY, leftIndent=10, rightIndent=10, spaceBefore=5, spaceAfter=5),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName=FONT, fontSize=9, leading=13, textColor=INK, leftIndent=14, firstLineIndent=-8, bulletIndent=4, spaceAfter=4),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName=MONO, fontSize=7.2, leading=9.3, textColor=INK, backColor=PALE, borderPadding=8, spaceBefore=4, spaceAfter=8),
        "center": ParagraphStyle("Center", parent=base["BodyText"], fontName=FONT, fontSize=8, leading=11, alignment=TA_CENTER, textColor=MUTED),
    }


def table(data, widths, *, header=True, font_size=7.5) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, PALE]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), BOLD),
        ]
    t.setStyle(TableStyle(style))
    return t


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(doc.leftMargin, 0.56 * inch, letter[0] - doc.rightMargin, 0.56 * inch)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.37 * inch, "AskATT System1 API appendix | Mark Austin")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.37 * inch, f"{doc.page}")
    canvas.restoreState()


def add_bullets(story, items, s) -> None:
    for item in items:
        story.append(p(f"- {item}", s["bullet"]))


def build() -> None:
    s = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.72 * inch,
        title="Appendix: AskATT System1 API",
        author="Mark Austin",
        subject="JEV-compatible Noul and Choice API implemented with GLiClass",
    )
    story = []

    story += [
        Spacer(1, 0.18 * inch),
        p("APPENDIX", ParagraphStyle("Kicker", parent=s["small"], fontName=BOLD, fontSize=10, textColor=ORANGE, tracking=1.5)),
        p("AskATT System1 API", s["title"]),
        p("A JEV-shaped Noul and Choice service implemented with GLiClass Modern Large v3", s["subtitle"]),
        Spacer(1, 0.12 * inch),
        Table([[p("INDEPENDENT IMPLEMENTATION", s["small"]), p("LOCKED TEST", s["small"]), p("SHARED PASS", s["small"])],
               [p("API compatibility, not a reproduction of JEV", s["callout"]), p("150 transcripts<br/>4,050 decisions", s["callout"]), p("27 labels in one GLiClass pass", s["callout"])]],
              colWidths=[2.55*inch, 1.55*inch, 2.25*inch],
              style=TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE_BLUE),("BOX",(0,0),(-1,-1),0.7,BLUE),("INNERGRID",(0,0),(-1,-1),0.3,colors.white),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)])),
        Spacer(1, 0.17 * inch),
        p("Executive result", s["h1"]),
        p("The service accepts one shared <b>state</b> plus named typed questions at <font name='AskMono'>POST /v1/systemone</font>. It compiles Noul and Choice questions into GLiClass labels and returns the familiar structured answer envelope. On the held-out test, Noul and conventional binary Choice each reached <b>94.10% accuracy, 96.07% recall, and 89.57% F1</b>.", s["body"]),
        p("This is an interface experiment, not a claim that GLiClass duplicates JEV's weights, calibration, confidence, limits, training method, or proprietary architecture.", s["callout"]),
        p("The current TypeSafe documentation states that JEV ingests the state once and evaluates questions in parallel. AskATT reaches a similar external shared-state pattern using GLiClass's uni-encoder path; the mechanisms are not asserted to be equivalent.", s["body"]),
        Spacer(1, 0.08 * inch),
        p("Mark Austin  |  September 20, 2026", s["small"]),
        PageBreak(),
    ]

    story += [p("Two users, two operating decisions", s["h1"])]
    persona = [
        [p("DATA SCIENTIST / APPLICATION TEAM", s["small"]), p("ENTERPRISE PLATFORM / ASK AT&amp;T", s["small"])],
        [p("Optimize for learning speed. Start with an API or local CPU checkpoint, preserve one request schema, calibrate per-label thresholds, and measure the workload before reserving hardware.", s["body"]),
         p("Optimize a multi-tenant portfolio. Pool stable base load, retain hosted burst/fallback capacity, version models and policies, and operate to p95/p99 latency rather than theoretical saturation.", s["body"])],
        [p("Bring evidence: label-level quality, state lengths, question counts, request rate, burstiness, privacy constraints, and latency SLO.", s["body"]),
         p("Provide isolation, auth, quotas, batching, autoscaling, observability, rollout controls, data-retention policy, and a funded reliability model.", s["body"])],
    ]
    story += [table(persona, [3.18*inch, 3.18*inch], font_size=8), Spacer(1, 10), p("When should a platform move off a usage-priced service?", s["h2"])]
    add_bullets(story, [
        "Quality parity is demonstrated on a labeled production shadow set.",
        "Sustained demand - not peak demand - exceeds the all-in cost crossover.",
        "The shared pool can keep accelerators usefully occupied across tenants.",
        "Measured p95/p99 latency meets the service objective with capacity headroom.",
        "Reliability, security, privacy, operations, and failure recovery are funded.",
    ], s)
    story += [
        Spacer(1, 4),
        p("Reference economics", s["h2"]),
        p("At 6,000 state tokens and 25 questions, the report's H100 proxy estimates <b>$0.051 per 1,000 states</b> for GLiClass at 100% paid utilization versus about <b>$0.352</b> for JEV Noul. The raw-inference crossover is 14.6% utilization. It is not a migration trigger: real self-hosting adds redundancy, orchestration, engineering, idle failover capacity, and support. A platform review at roughly 25-40% sustained utilization is a more prudent starting point, followed by an all-in measured business case.", s["body"]),
        p("Hybrid default: local shared pool for proven base load; hosted JEV or another managed service for bursts, fallback, long contexts, and labels that miss the local quality target.", s["callout"]),
        PageBreak(),
    ]

    story += [p("How the compatibility layer works", s["h1"]), Architecture(), Spacer(1, 6)]
    story += [
        p("Noul", s["h2"]), p("The positive criterion - or the instructions when no positive criterion is supplied - becomes one GLiClass label. The returned score is the Noul probability.", s["body"]),
        p("Binary Choice", s["h2"]), p("For present/absent, yes/no, or true/false options, the positive score is <font name='AskMono'>p</font> and the other option is <font name='AskMono'>1-p</font>. This makes Noul and binary Choice intentionally equivalent at a 0.50 selection threshold.", s["body"]),
        p("General Choice", s["h2"]), p("Each option becomes a label. The adapter applies a within-question softmax over GLiClass score log-odds. Confidence is normalized distribution concentration; it is not TypeSafe's confidence calculation.", s["body"]),
        p("Pass count", s["h2"]), p("Up to 64 compiled labels run in one shared-state pass by default. Additional label groups repeat the state. A multi-option Choice may therefore consume multiple compiled-label slots even though it is one question object.", s["body"]),
        PageBreak(),
    ]

    compatibility = [
        ["Capability", "TypeSafe JEV", "AskATT implementation"],
        ["Endpoint", "POST /v1/systemone", "Same path"],
        ["State", "String, object, or array", "Same shapes; serialized to text"],
        ["Noul / Choice", "Both; Choice up to 255 options", "Both; Choice up to 255 options"],
        ["Score", "Supported", "Not implemented; HTTP 422"],
        ["Context", "64k total; state + longest question <=32k", "8,192-token GLiClass input"],
        ["Question cap", "Token/rate limits govern", "64 question objects by default"],
        ["Confidence", "JEV-defined", "Normalized entropy proxy"],
        ["Usage", "JEV billable tokens", "Serialized GLiClass input tokens"],
        ["Auth", "Bearer token", "Optional bearer token"],
        ["Model field", "Selects alias/version", "Required; local configured model answers"],
    ]
    story += [
        p("Compatibility is deliberately bounded", s["h1"]),
        table(compatibility, [1.25*inch, 2.35*inch, 2.75*inch], font_size=7.2),
        Spacer(1, 8),
        p("The service exposes extra diagnostic headers: inference milliseconds, forward-pass count, compiled-label count, and truncation status. These make state repetition and context loss observable to the caller.", s["body"]),
        p("Sources", s["h2"]),
        p("TypeSafe API reference: https://docs.typesafe.ai/api<br/>TypeSafe models and limits: https://docs.typesafe.ai/models<br/>Accessed September 20, 2026.", s["small"]),
        PageBreak(),
    ]

    results = [
        ["Measure", "Noul", "Binary Choice"],
        ["Accuracy", "94.10%", "94.10%"],
        ["Precision", "83.89%", "83.89%"],
        ["Recall", "96.07%", "96.07%"],
        ["F1", "89.57%", "89.57%"],
        ["Exact transcript match", "16.67%", "16.67%"],
        ["TP / FP / FN / TN", "1,026 / 197 / 42 / 2,785", "Same"],
        ["Input tokens", "78,219", "78,219"],
        ["Forward passes/state", "1.0", "1.0"],
        ["Truncated rows", "0", "0"],
        ["Local CPU wall time", "78.62 s", "83.10 s"],
        ["Local CPU states/s", "1.91", "1.81"],
    ]
    story += [
        p("Test results", s["h1"]),
        p("Six deterministic contract tests and one opt-in real-checkpoint integration test passed. The full benchmark used the locked 150-transcript test split with 27 attributes per transcript: 4,050 binary decisions.", s["body"]),
        table(results, [2.15*inch, 2.10*inch, 2.10*inch], font_size=7.5),
        Spacer(1, 8),
        p("The Noul and binary Choice predictions were identical. Their small CPU timing difference is run-to-run overhead rather than different model work.", s["body"]),
        p("H100 capacity estimate", s["h2"]),
        p("Using the report's 170,331 processed-token/s H100 proxy and $5/GPU-hour assumption, the benchmark's 521.46 serialized tokens imply approximately <b>3.06 ms of saturated GPU resource time per state</b> and <b>$0.00425 per 1,000 states</b> at 100% utilization. This is not measured end-to-end request latency.", s["callout"]),
        PageBreak(),
    ]

    request_code = '''curl http://127.0.0.1:8000/v1/systemone \\\n+  -H 'Content-Type: application/json' \\\n+  -d '{
    "model": "jev-latest",
    "state": "The caller disputes an unfamiliar fee.",
    "questions": {
      "unknown_charge": {
        "type": "noul",
        "instructions": "Is an unknown charge present?",
        "criteria": {"true": "Unknown charges on the bill"}
      },
      "department": {
        "type": "choice",
        "instructions": "Route the issue",
        "criteria": {
          "billing": "Billing, charges, and payments",
          "technical": "Internet or device failure",
          "sales": "New purchases or upgrades"
        }
      }
    }
  }' '''
    story += [
        p("Calling the API", s["h1"]),
        p(request_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"), s["code"]),
        p("Representative response", s["h2"]),
        p('''{
  "model": "askatt-gliclass-modern-large-v3.0",
  "answers": {
    "unknown_charge": {"type": "noul", "noul": 0.7835},
    "department": {
      "type": "choice", "choice": "billing",
      "probabilities": {"billing": 0.9890, "technical": 0.0067,
                        "sales": 0.0043},
      "confidence": 0.932
    }
  },
  "usage": {"input_tokens": 61, "output_tokens": 0}
}'''.replace("\n", "<br/>"), s["code"]),
        p("The response uses JEV field names while truthfully identifying the configured AskATT model. The request's <font name='AskMono'>model</font> is required for schema compatibility but does not dynamically load a checkpoint.", s["body"]),
        PageBreak(),
    ]

    formulas = '''passes = ceil(compiled_labels / labels_per_pass)
processed_tokens ~= passes * state_tokens + serialized_label_tokens
resource_seconds/state = processed_tokens / saturated_tokens_per_second
cost/1,000 states = 1,000 * resource_seconds/state * GPU_dollars/hour
                    / (3,600 * paid_utilization)

JEV billable_tokens ~= state_tokens + sum(question_tokens) + overhead
JEV cost/1,000 = 1,000 * billable_tokens * dollars/input_token

end_to_end_latency = queueing + preprocessing + model_service
                     + postprocessing + network'''
    story += [
        p("Cost, utilization, and latency", s["h1"]),
        p(formulas.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"), s["code"]),
        p("The formula makes the platform trade-off visible. At low paid utilization, idle GPU hours dominate. As utilization approaches saturation, cost per state falls but queueing grows sharply. Offline jobs can run near saturation; interactive services need headroom and should be governed by measured percentile latency.", s["body"]),
        p("Reproduce", s["h2"]),
        p('''python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r askatt_system1_api/requirements.txt
export ASKATT_GLICLASS_MODEL=models/gliclass-modern-large-v3.0
export ASKATT_DEVICE=cpu
.venv/bin/python -m askatt_system1_api

.venv/bin/python -m pytest -q tests/test_askatt_system1_api.py
ASKATT_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -q tests/test_askatt_system1_model.py
.venv/bin/python scripts/benchmark_askatt_system1_api.py'''.replace("\n", "<br/>"), s["code"]),
        p("Production checklist", s["h2"]),
    ]
    add_bullets(story, [
        "Version prompts, criteria, thresholds, and model checkpoints together.",
        "Benchmark CUDA with representative concurrency, state lengths, and question counts.",
        "Add dynamic batching, back-pressure, tenant-scoped auth, quotas, and audit logs.",
        "Canary against pinned JEV versions and a labeled production shadow set.",
        "Route proven base load locally; retain a hosted fallback until reliability and quality are established.",
    ], s)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
