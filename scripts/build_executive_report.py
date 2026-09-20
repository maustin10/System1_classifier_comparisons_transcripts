#!/usr/bin/env python3
"""Create a concise executive PDF from the benchmark's decision charts."""
from __future__ import annotations

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas

from build_report import (
    BLUE,
    MUTED,
    NAVY,
    ORANGE,
    PAGE_H,
    PAGE_W,
    PALE,
    PURPLE,
    ROOT,
    bullet_list,
    chart_page,
    footer,
    page_title,
    wrapped,
)

OUTPUT = ROOT / "report" / "transcript-classifier-executive-summary.pdf"


def cover(c: canvas.Canvas) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(PURPLE)
    c.rect(0, PAGE_H - 13, PAGE_W * 0.62, 13, fill=1, stroke=0)
    c.setFillColor(ORANGE)
    c.rect(PAGE_W * 0.62, PAGE_H - 13, PAGE_W * 0.38, 13, fill=1, stroke=0)

    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 33)
    c.drawString(46, PAGE_H - 112, "Executive classifier decision guide")
    c.setFont("Helvetica-Oblique", 15)
    c.setFillColor(HexColor("#C8D3E3"))
    c.drawString(47, PAGE_H - 148, "State length × question count × quality × cost")

    cards = [
        ("Fixed labels", "ModernBERT trained", "lowest modeled cost", BLUE),
        ("Runtime labels", "GLiClass Modern", "self-hosted flexibility", ORANGE),
        ("Managed API", "JEV Noul", "quality / cost balance", PURPLE),
    ]
    x = 47
    for heading, model, detail, color in cards:
        c.setFillColor(HexColor("#22304F"))
        c.roundRect(x, 184, 210, 108, 8, fill=1, stroke=0)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x + 17, 255, heading)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x + 17, 226, model)
        c.setFillColor(HexColor("#C8D3E3"))
        c.setFont("Helvetica", 9.5)
        c.drawString(x + 17, 203, detail)
        x += 225

    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(47, 82, "Mark Austin")
    c.setFillColor(HexColor("#C8D3E3"))
    c.setFont("Helvetica", 9.5)
    c.drawString(47, 62, "Independent synthetic evaluation · September 20, 2026")
    c.showPage()


def recommendation_page(c: canvas.Canvas, page_number: int) -> None:
    page_title(c, "What should an executive choose?", "Select the operating model before selecting the model")

    recommendations = [
        ("Known, stable taxonomy", "ModernBERT trained heads", "Best modeled economics with 99.16% test accuracy. New questions require labeled data and retraining."),
        ("Runtime labels, self-hosted", "GLiClass Modern", "Best open zero-shot trade-off tested: 97.90% calibrated accuracy and near-trained-encoder modeled cost."),
        ("Runtime questions, managed service", "JEV Noul", "99.90% calibrated accuracy with shared-state-like billing and substantially lower modeled cost than the general LLMs."),
        ("Broad reasoning or maximum test score", "Luna or Sol", "Near-ceiling quality, but use when semantic breadth justifies the premium. Sol's advantage over calibrated JEV was one decision in 4,050."),
    ]

    y = PAGE_H - 104
    colors = [BLUE, ORANGE, PURPLE, HexColor("#303846")]
    for (use_case, choice, rationale), color in zip(recommendations, colors):
        c.setFillColor(PALE)
        c.roundRect(44, y - 83, 704, 73, 7, fill=1, stroke=0)
        c.setFillColor(color)
        c.rect(44, y - 83, 7, 73, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(63, y - 31, use_case)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(340, y - 31, choice)
        wrapped(c, rationale, 63, y - 47, 661, font_size=9.2, leading=11.5)
        y -= 86

    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(44, 114, "Executive guardrails")
    bullet_list(c, [
        "Quality results come from a synthetic, explicit-evidence test and are not production guarantees.",
        "Cost curves are modeled from token shapes, published/API rates, and H100 throughput proxies.",
        "Validate with real states, real questions, target hardware, and operational error costs before procurement.",
    ], 44, 94, 704, font_size=8.8)
    footer(c, page_number)
    c.showPage()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Executive classifier decision guide")
    c.setAuthor("Mark Austin")
    c.setSubject("Executive comparison of fixed encoders, runtime-label classifiers, JEV, and LLM classifiers")
    cover(c)
    chart_page(c, 2, "executive-decision-matrix.png")
    chart_page(c, 3, "executive-state-question-cost-bars.png")
    recommendation_page(c, 4)
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
