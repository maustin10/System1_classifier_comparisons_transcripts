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
    c.drawString(36, 13, "Independent synthetic transcript classifier benchmark · Mark Austin · September 19, 2026")
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
    c.drawString(47, 48, "Independent evaluation · September 19, 2026")
    c.showPage()


def executive_page(c: canvas.Canvas, page_number: int) -> None:
    page_title(c, "Executive findings", "Quality, calibration, and operating-cost implications")
    findings = [
        "Sol achieved the highest measured quality: 99.93% accuracy and 99.86% micro F1, with three errors across 4,050 held-out decisions.",
        "Validation-calibrated JEV Noul was one decision behind Sol: 99.90% accuracy and 99.81% F1. Its estimated recurring API charge is about $0.117 per 1,000 transcripts under the measured prompt shape.",
        "The strongest fully local result was frozen ModernBERT plus 27 supervised logistic heads: 99.16% accuracy and 98.41% F1.",
        "Threshold calibration alone improved unchanged zero-shot ModernBERT from 92.96% to 96.42% accuracy and cut errors from 285 to 145.",
        "DeBERTa-v3-large zero-shot (-c) did not beat ModernBERT zero-shot and was approximately 2.47x slower on the measured CPU path.",
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


def conclusions_page(c: canvas.Canvas, page_number: int) -> None:
    page_title(c, "Conclusions, limitations, and sources")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(45, PAGE_H - 95, "Practical interpretation")
    y = bullet_list(c, [
        "For the highest measured quality, Sol led by one decision over calibrated JEV Noul.",
        "For low estimated API cost with near-frontier quality, calibrated JEV Noul was the standout in this synthetic task.",
        "For a fully local and controllable system, trained ModernBERT was strong, but it required labeled training data.",
        "Threshold calibration is worthwhile, but it does not replace supervised heads when enough labeled examples exist.",
    ], 45, PAGE_H - 118, 702, font_size=9.5)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(45, y - 7, "Limitations")
    y = bullet_list(c, [
        "Synthetic templates are more explicit and regular than production conversations.",
        "Only 150 transcripts were held out; attribute decisions within a transcript are correlated.",
        "Choice versus Noul changed multiple factors and is not a clean primitive-only comparison.",
        "Latency paths and hardware differed; cost figures are measured or estimated as labeled, not full TCO.",
        "Sol/Luna standardized cost estimates exclude unavailable hidden reasoning-token usage.",
    ], 45, y - 30, 702, font_size=9.1)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(45, y - 3, "Sources")
    sources = [
        "TypeSafe API: https://docs.typesafe.ai/api",
        "TypeSafe JEV launch/pricing: https://typesafe.ai/blog/introducing-system-one-models-and-jev",
        "OpenAI Sol pricing: https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "OpenAI Luna pricing: https://openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6/",
        "Model cards: https://huggingface.co/MoritzLaurer",
    ]
    y = bullet_list(c, sources, 45, y - 25, 702, font_size=8.2)
    footer(c, page_number)
    c.showPage()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("System1 classifier comparisons for customer-care transcripts")
    c.setAuthor("Mark Austin")
    c.setSubject("Independent comparison of ModernBERT, DeBERTa, TypeSafe.ai JEV, GPT-5.6 Sol, and GPT-5.6 Luna")
    cover(c)
    executive_page(c, 2)
    chart_page(c, 3, "f1-comparison.png")
    chart_page(c, 4, "accuracy-comparison.png")
    chart_page(c, 5, "all-metrics-table.png")
    chart_page(c, 6, "estimated-cost-1000-transcripts.png")
    methodology_page(c, 7)
    conclusions_page(c, 8)
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
