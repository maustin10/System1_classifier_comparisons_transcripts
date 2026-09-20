#!/usr/bin/env python3
"""Create the landscape PDF report and verify its basic structure."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "transcript-classifier-comparison.pdf"
SUMMARY = json.loads((ROOT / "data" / "summary_metrics.json").read_text())

PAGE_W, PAGE_H = landscape(letter)
NAVY = HexColor("#17233F")
BLUE = HexColor("#0B678B")
PURPLE = HexColor("#8059C3")
ORANGE = HexColor("#D47900")
MUTED = HexColor("#66758A")
PALE = HexColor("#EEF3F7")
GRID = HexColor("#D8E0E8")


def footer(c: canvas.Canvas, page_number: int) -> None:
    c.setStrokeColor(GRID)
    c.line(36, 25, PAGE_W - 36, 25)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(36, 13, "Independent synthetic transcript classifier benchmark · Mark Austin · September 20, 2026")
    c.drawRightString(PAGE_W - 36, 13, str(page_number))


def page_title(c: canvas.Canvas, title: str, subtitle: str = "") -> None:
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 23)
    c.drawString(40, PAGE_H - 50, title)
    if subtitle:
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Oblique", 10.5)
        c.drawString(40, PAGE_H - 69, subtitle)


def wrapped(c: canvas.Canvas, text: str, x: float, y: float, width: float, font_size: float = 10, leading: float = 13, color=NAVY, bold: bool = False) -> float:
    style = ParagraphStyle(
        "body",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=font_size,
        leading=leading,
        textColor=color,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, PAGE_H)
    paragraph.drawOn(c, x, y - height)
    return y - height


def bullet_list(c: canvas.Canvas, items: list[str], x: float, y: float, width: float, font_size: float = 9.5) -> float:
    for item in items:
        y = wrapped(c, f"• {item}", x, y, width, font_size=font_size, leading=font_size + 3, color=NAVY)
        y -= 5
    return y


def chart_page(c: canvas.Canvas, page_number: int, image_name: str) -> None:
    path = ROOT / "charts" / image_name
    with PILImage.open(path) as image:
        image_w, image_h = image.size
    max_w, max_h = PAGE_W - 34, PAGE_H - 42
    scale = min(max_w / image_w, max_h / image_h)
    draw_w, draw_h = image_w * scale, image_h * scale
    x, y = (PAGE_W - draw_w) / 2, 29 + (max_h - draw_h) / 2
    c.drawImage(ImageReader(str(path)), x, y, draw_w, draw_h, preserveAspectRatio=True, mask="auto")
    footer(c, page_number)
    c.showPage()


def cover(c: canvas.Canvas) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(PURPLE)
    c.rect(0, PAGE_H - 13, PAGE_W * 0.62, 13, fill=1, stroke=0)
    c.setFillColor(ORANGE)
    c.rect(PAGE_W * 0.62, PAGE_H - 13, PAGE_W * 0.38, 13, fill=1, stroke=0)

    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 31)
    c.drawString(46, PAGE_H - 104, "System1 classifier comparisons")
    c.drawString(46, PAGE_H - 142, "for customer-care transcripts")
    c.setFont("Helvetica-Oblique", 14)
    c.setFillColor(HexColor("#C8D3E3"))
    c.drawString(47, PAGE_H - 174, "Open encoders · TypeSafe.ai JEV · GPT-5.6 Sol and Luna")

    cards = [
        ("99.93%", "Best accuracy", "GPT-5.6 Sol", ORANGE),
        ("99.90%", "JEV calibrated", "4 / 4,050 errors", PURPLE),
        ("99.16%", "Best local encoder", "ModernBERT trained", BLUE),
    ]
    x = 47
    for value, label, detail, color in cards:
        c.setFillColor(HexColor("#22304F"))
        c.roundRect(x, 165, 210, 112, 8, fill=1, stroke=0)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 25)
        c.drawString(x + 17, 235, value)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x + 17, 209, label)
        c.setFillColor(HexColor("#C8D3E3"))
        c.setFont("Helvetica", 9.5)
        c.drawString(x + 17, 187, detail)
        x += 225

    c.setFillColor(HexColor("#C8D3E3"))
    c.setFont("Helvetica", 10)
    c.drawString(47, 100, "1,000 synthetic conversations · 27 attributes · 700 train / 150 validation / 150 locked test")
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(white)
    c.drawString(47, 65, "Mark Austin")
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#C8D3E3"))
    c.drawString(47, 48, "Independent evaluation · September 20, 2026")
    c.showPage()


def executive_page(c: canvas.Canvas, page_number: int) -> None:
    page_title(c, "Executive findings", "Quality, calibration, and operating-cost implications")
    findings = [
        "Sol achieved the highest measured quality: 99.93% accuracy and 99.86% micro F1, with three errors across 4,050 held-out decisions.",
        "Validation-calibrated JEV Noul was one decision behind Sol: 99.90% accuracy and 99.81% F1. Its estimated recurring API charge is about $0.117 per 1,000 transcripts under the measured prompt shape.",
        "The strongest fully local result was frozen ModernBERT plus 27 supervised logistic heads: 99.16% accuracy and 98.41% F1.",
        "Threshold calibration alone improved unchanged zero-shot ModernBERT from 92.96% to 96.42% accuracy and cut errors from 285 to 145.",
        "GLiClass Modern Large v3 was the strongest zero-shot open encoder: validation-only calibration reached 97.90% accuracy and 96.00% F1.",
        "DeBERTa-v3-large zero-shot (-c) did not beat ModernBERT zero-shot and was approximately 2.47x slower on the measured CPU path.",
        "At the 224-token benchmark mean, simulated cost per 1,000 transcripts is $0.0018 for trained ModernBERT, $0.0043 for GLiClass Modern, $0.0525 for zero-shot ModernBERT, and $0.1169 for JEV Noul.",
        "The standardized one-call API estimates are $0.4000 for Luna and $7.1520 for Sol per 1,000 benchmark-mean transcripts; hidden reasoning tokens are excluded.",
    ]
    y = bullet_list(c, findings, 48, PAGE_H - 100, 690, font_size=11)

    c.setFillColor(PALE)
    c.roundRect(48, 58, 696, 102, 7, fill=1, stroke=0)
    wrapped(
        c,
        "<b>Interpret carefully.</b> This is a synthetic, template-driven smoke test. It is useful for controlled comparisons and pipeline validation, but it does not establish production accuracy. The JEV Choice and Noul runs also differ in state structure and criteria specificity, so their gap is not a primitive-only A/B result.",
        65,
        137,
        662,
        font_size=10.5,
        leading=14,
    )
    footer(c, page_number)
    c.showPage()


def methodology_page(c: canvas.Canvas, page_number: int) -> None:
    page_title(c, "Methodology and controls", "What was held fixed and what changed")
    left_x, right_x, col_w = 45, 410, 337
    y_left = PAGE_H - 100
    y_right = y_left

    sections_left = [
        ("Dataset", [
            "1,000 synthetic alternating Agent/Caller conversations.",
            "27 binary attributes per transcript.",
            "Scenario-family-stratified split: 700 / 150 / 150.",
            "Test truth excluded from training, prompt refinement, and threshold selection.",
        ]),
        ("Metrics", [
            "Micro accuracy, precision, recall, and F1 over 4,050 decisions.",
            "Exact match requires all 27 labels in a transcript to be correct.",
        ]),
    ]
    sections_right = [
        ("Encoder variants", [
            "Zero-shot: fixed premise/hypothesis NLI probabilities.",
            "GLiClass: uni-encoder label scoring; Modern v3 used one pass, while the 512-token Large v3 used two label groups to prevent truncation.",
            "Optimized threshold: 27 validation-selected scalar cutoffs; no weight training.",
            "Trained: frozen 1,024-d embedding plus 27 logistic heads and validation thresholds.",
        ]),
        ("JEV and LLM variants", [
            "JEV sends 27 typed questions in one request per transcript.",
            "Final Noul uses structured speakers and strict true/false boundaries.",
            "Sol and Luna produced blind binary JSON predictions before truth was joined.",
        ]),
    ]
    for heading, items in sections_left:
        c.setFillColor(BLUE)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(left_x, y_left, heading)
        y_left = bullet_list(c, items, left_x, y_left - 22, col_w, font_size=9.5) - 10
    for heading, items in sections_right:
        c.setFillColor(PURPLE)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(right_x, y_right, heading)
        y_right = bullet_list(c, items, right_x, y_right - 22, col_w, font_size=9.5) - 10

    c.setFillColor(PALE)
    c.roundRect(45, 56, 702, 83, 7, fill=1, stroke=0)
    wrapped(
        c,
        "<b>Key distinction:</b> trained ModernBERT learned approximately 27,675 logistic-head parameters from 700 rows. The optimized-threshold variant learned only 27 cutoffs from validation. The difference between their test results therefore estimates the value of supervised decision boundaries beyond operating-point calibration.",
        62,
        118,
        668,
        font_size=9.5,
        leading=12.5,
    )
    footer(c, page_number)
    c.showPage()


def operating_model_page(c: canvas.Canvas, page_number: int) -> None:
    page_title(c, "What is shared - and what is fixed", "The low-cost trained encoder and arbitrary-question JEV solve different serving problems")
    left_x, right_x, col_w = 45, 410, 337

    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left_x, PAGE_H - 105, "ModernBERT paths")
    y = wrapped(
        c,
        "<b>Zero-shot NLI:</b> N state/question pairs. Measured token amplification: 28.67x for 27 questions.",
        left_x,
        PAGE_H - 132,
        col_w,
        font_size=11,
        leading=15,
    )
    y = bullet_list(c, [
        "It accepts arbitrary hypotheses, but repeats the state inside each premise/hypothesis sequence.",
        "Trained heads encode the state once, then apply 27 logistic functions.",
        "The trained path has no question tokens at inference; label meaning is stored in fixed head weights.",
        "Above about 8.2k state tokens, both paths require overlapping chunks and probability aggregation.",
        "A novel question requires a new or retrained head.",
    ], left_x, y - 18, col_w, font_size=9.5)

    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(right_x, PAGE_H - 105, "JEV evidence")
    y2 = bullet_list(c, [
        "The API accepts arbitrary state and arbitrary typed questions together.",
        "A state can approach 32k tokens minus its longest question; larger states require multiple requests.",
        "Billing behaves like state + N x question + fixed overhead.",
        "Latency was approximately flat from 1 to 16 questions.",
        "TypeSafe claims a new architecture, parallel sampler, and RLCD training.",
        "Public materials do not disclose the compute graph or prove state is encoded once.",
    ], right_x, PAGE_H - 132, col_w, font_size=9.5)

    c.setFillColor(PALE)
    c.roundRect(45, 59, 702, 106, 7, fill=1, stroke=0)
    wrapped(
        c,
        "<b>Conclusion:</b> JEV's arbitrary-question, shared-state-like product behavior is real and useful. It is not enough to verify architectural novelty. Ordinary GPU batching can hide repeated state compute at small N, and known dual-encoder, cached-state, late-interaction, or shared-memory designs could expose similar behavior.",
        62,
        140,
        668,
        font_size=10.2,
        leading=13.5,
    )
    footer(c, page_number)
    c.showPage()


def cost_formulas_page(c: canvas.Canvas, page_number: int) -> None:
    page_title(c, "Cost formulas and assumptions", "Estimated USD per 1,000 transcripts")

    c.setFillColor(PALE)
    c.roundRect(44, PAGE_H - 178, 704, 88, 7, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(60, PAGE_H - 116, "Definitions")
    c.setFont("Courier", 7.8)
    c.drawString(60, PAGE_H - 135, "S = state tokens      q = 27 questions / labels      overlap = 256 tokens")
    c.drawString(60, PAGE_H - 151, "kM  = max(1, ceil((S - 256)/(8178.11 - 256)))")
    c.drawString(400, PAGE_H - 151, "kGM = max(1, ceil((S - 256)/(7895 - 256)))")
    c.drawString(60, PAGE_H - 167, "kGL = max(1, ceil((S - 256)/(357 - 256)))")
    c.drawString(400, PAGE_H - 167, "kJ  = max(1, ceil((S - 256)/(31890 - 256)))")

    formulas = [
        (
            "ModernBERT trained heads",
            "0.008154 * [S + 256*(kM - 1) + 2*kM] / 1,000",
            BLUE,
        ),
        (
            "GLiClass Modern Large",
            "0.008154 * [S + 256*(kGM - 1) + 297*kGM] / 1,000",
            ORANGE,
        ),
        (
            "GLiClass Large (2 groups)",
            "0.012669 * {2*[S + 256*(kGL - 1)] + 294*kGL} / 1,000",
            HexColor("#C44E52"),
        ),
        (
            "ModernBERT zero-shot",
            "0.008154 * {27*[S + 256*(kM - 1)] + 375*kM} / 1,000",
            HexColor("#2A9D8F"),
        ),
        (
            "JEV Noul",
            "0.042 * [S + 256*(kJ - 1) + 2559.74*kJ] / 1,000",
            PURPLE,
        ),
        (
            "JEV Choice",
            "0.042 * [S + 256*(kJ - 1) + 2963.67*kJ] / 1,000",
            HexColor("#A78BDB"),
        ),
        (
            "GPT-5.6 Luna",
            "[0.20*(S + 503.53) + 1.20*212] / 1,000",
            HexColor("#F4A261"),
        ),
        (
            "GPT-5.6 Sol",
            "[4.00*(S + 503.53) + 20.00*212] / 1,000",
            HexColor("#303846"),
        ),
    ]
    y = PAGE_H - 205
    for label, formula, color in formulas:
        c.setStrokeColor(GRID)
        c.setFillColor(white)
        c.roundRect(44, y - 25, 704, 29, 6, fill=1, stroke=1)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 8.1)
        c.drawString(60, y - 8, label)
        c.setFillColor(NAVY)
        c.setFont("Courier", 7.3)
        c.drawString(250, y - 8, formula)
        y -= 30

    c.setFillColor(PALE)
    c.roundRect(44, 44, 704, 94, 7, fill=1, stroke=0)
    wrapped(
        c,
        "<b>Rates.</b> ModernBERT and GLiClass Modern use the 170.3k processed-token/s H100 proxy. JEV uses USD 0.042/M input. Luna uses USD 0.20/M input and USD 1.20/M output; Sol uses USD 4/M input and USD 20/M output.",
        60,
        124,
        672,
        font_size=8.3,
        leading=10.3,
    )
    wrapped(
        c,
        "<b>Assumptions.</b> GLiClass Large uses a 109.6k token/s proxy. Sol/Luna use S + 503.53 input tokens and 212 output tokens; hidden reasoning tokens are excluded. Neither GLiClass checkpoint was measured on H100 here.",
        60,
        80,
        672,
        font_size=8.3,
        leading=10.3,
    )
    footer(c, page_number)
    c.showPage()


def conclusions_page(c: canvas.Canvas, page_number: int) -> None:
    page_title(c, "Conclusions, limitations, and sources")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(45, PAGE_H - 95, "Practical interpretation")
    y = bullet_list(c, [
        "For the highest measured quality, Sol led by one decision over calibrated JEV Noul, but its standardized benchmark-mean API estimate is $7.152 per 1,000 transcripts versus $0.400 for Luna and $0.117 for JEV Noul.",
        "GLiClass Modern combines arbitrary runtime labels with a simulated $0.0043 cost per 1,000 benchmark-mean transcripts, versus $0.0525 for zero-shot ModernBERT and $0.1169 for JEV Noul.",
        "Trained ModernBERT was by far the cheapest path, but only because its 27 questions were converted into fixed supervised heads.",
        "GLiClass Modern Large improved the calibrated zero-shot open-encoder result to 97.90% accuracy, but remained below trained ModernBERT.",
        "Threshold calibration is worthwhile, but it does not replace supervised heads when enough labeled examples exist.",
    ], 45, PAGE_H - 118, 702, font_size=9.5)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(45, y - 7, "Limitations")
    y = bullet_list(c, [
        "Synthetic templates are more explicit and regular than production conversations.",
        "Only 150 transcripts were held out; attribute decisions within a transcript are correlated.",
        "Choice versus Noul changed multiple factors and is not a clean primitive-only comparison.",
        "The H100 scenario uses throughput proxies and holds token throughput constant across state lengths; GLiClass Modern inherits the ModernBERT-large proxy and GLiClass Large scales it using an official A6000 relative-throughput ratio.",
        "JEV cost is extrapolated from billing; hosted capacity at the equivalent throughput was not tested.",
        "Sol/Luna standardized cost estimates exclude unavailable hidden reasoning-token usage.",
        "Sol/Luna state scaling assumes one added LLM input token per ModernBERT state-axis token; tokenizer differences make this approximate.",
        "GLiClass Large used two label groups while GLiClass Modern Large used one; their local timings are not a one-pass checkpoint comparison.",
    ], 45, y - 30, 702, font_size=9.1)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(45, y - 3, "Sources")
    sources = [
        "TypeSafe API: https://docs.typesafe.ai/api",
        "TypeSafe model limits: https://docs.typesafe.ai/models",
        "TypeSafe JEV launch/pricing: https://typesafe.ai/blog/introducing-system-one-models-and-jev",
        "ModernBERT documentation: https://huggingface.co/docs/transformers/en/model_doc/modernbert",
        "H100 ModernBERT-base observation: https://www.linkedin.com/posts/michael-feil_the-latest-release-of-infinity-httpslnkdin-activity-7280971190632943616-E07N",
        "OpenAI Sol pricing: https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "OpenAI Luna pricing: https://developers.openai.com/api/docs/models",
        "GLiClass Modern Large v3: https://huggingface.co/knowledgator/gliclass-modern-large-v3.0",
        "GLiClass Large v3: https://huggingface.co/knowledgator/gliclass-large-v3.0",
    ]
    y = bullet_list(c, sources, 45, y - 25, 702, font_size=7.9)
    footer(c, page_number)
    c.showPage()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("System1 classifier comparisons for customer-care transcripts")
    c.setAuthor("Mark Austin")
    c.setSubject("Independent comparison of ModernBERT, DeBERTa, GLiClass, TypeSafe.ai JEV, GPT-5.6 Sol, and GPT-5.6 Luna")
    cover(c)
    executive_page(c, 2)
    chart_page(c, 3, "f1-comparison.png")
    chart_page(c, 4, "accuracy-comparison.png")
    chart_page(c, 5, "all-metrics-table.png")
    chart_page(c, 6, "estimated-cost-1000-transcripts.png")
    chart_page(c, 7, "normalized-cost-vs-state-tokens.png")
    cost_formulas_page(c, 8)
    operating_model_page(c, 9)
    methodology_page(c, 10)
    conclusions_page(c, 11)
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
