#!/usr/bin/env python3
"""Render the classifier-tokenomics Markdown outline and all figures as a PDF."""
from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "articles" / "blog-outline-classifier-tokenomics.md"
OUTPUT = ROOT / "output" / "pdf" / "inside-classifier-tokenomics-blog-outline.pdf"
PAGE = landscape(letter)

NAVY = colors.HexColor("#172554")
INK = colors.HexColor("#172033")
BLUE = colors.HexColor("#075985")
TEAL = colors.HexColor("#0F766E")
PURPLE = colors.HexColor("#7C3AED")
ORANGE = colors.HexColor("#C76F00")
GREEN = colors.HexColor("#15803D")
MUTED = colors.HexColor("#64748B")
PALE = colors.HexColor("#F1F5F9")
PALE_BLUE = colors.HexColor("#E0F2FE")
LINE = colors.HexColor("#CBD5E1")


def register_fonts() -> tuple[str, str, str]:
    normal_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    mono_candidates = [
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]
    normal = next((item for item in normal_candidates if item.exists()), None)
    bold = next((item for item in bold_candidates if item.exists()), None)
    mono = next((item for item in mono_candidates if item.exists()), None)
    if normal and bold and mono:
        pdfmetrics.registerFont(TTFont("BlogSans", str(normal)))
        pdfmetrics.registerFont(TTFont("BlogSansBold", str(bold)))
        pdfmetrics.registerFont(TTFont("BlogMono", str(mono)))
        return "BlogSans", "BlogSansBold", "BlogMono"
    return "Helvetica", "Helvetica-Bold", "Courier"


FONT, BOLD, MONO = register_fonts()


def clean_ascii(value: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2011": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00d7": "x",
        "\u2248": "approximately ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def markup(value: str) -> str:
    value = html.escape(clean_ascii(value), quote=False)
    value = re.sub(r"`([^`]+)`", rf'<font name="{MONO}" color="#334155">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle("CoverTitle", parent=base["Title"], fontName=BOLD, fontSize=27, leading=32, textColor=NAVY, alignment=TA_LEFT, spaceAfter=15),
        "cover_sub": ParagraphStyle("CoverSub", parent=base["Normal"], fontName=FONT, fontSize=13, leading=19, textColor=MUTED, spaceAfter=12),
        "kicker": ParagraphStyle("Kicker", parent=base["Normal"], fontName=BOLD, fontSize=9.5, leading=12, textColor=ORANGE, tracking=1.4, spaceAfter=9),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName=BOLD, fontSize=20, leading=24, textColor=NAVY, spaceBefore=10, spaceAfter=9, keepWithNext=True),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName=BOLD, fontSize=15.5, leading=19, textColor=BLUE, spaceBefore=12, spaceAfter=7, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName=FONT, fontSize=9.4, leading=13.8, textColor=INK, rightIndent=95, spaceAfter=7),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName=FONT, fontSize=9.2, leading=13, textColor=INK, leftIndent=17, firstLineIndent=-9, bulletIndent=5, rightIndent=95, spaceAfter=4),
        "number": ParagraphStyle("Number", parent=base["BodyText"], fontName=FONT, fontSize=9.2, leading=13, textColor=INK, leftIndent=20, firstLineIndent=-14, rightIndent=95, spaceAfter=4),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName=MONO, fontSize=8.1, leading=11.2, textColor=INK, backColor=PALE, borderPadding=9, leftIndent=10, rightIndent=120, spaceBefore=5, spaceAfter=9),
        "figure_label": ParagraphStyle("FigureLabel", parent=base["Normal"], fontName=BOLD, fontSize=9, leading=12, textColor=ORANGE, tracking=0.8, spaceAfter=5),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName=FONT, fontSize=8.2, leading=11, textColor=MUTED, alignment=TA_CENTER, leftIndent=22, rightIndent=22, spaceBefore=5),
        "toc": ParagraphStyle("TOC", parent=base["BodyText"], fontName=FONT, fontSize=10.5, leading=15, textColor=INK, leftIndent=16, firstLineIndent=-16, spaceAfter=7),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName=FONT, fontSize=7.5, leading=10, textColor=MUTED),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName=BOLD, fontSize=12, leading=17, textColor=NAVY, leftIndent=12, rightIndent=12, spaceAfter=4),
    }


class ReferenceArchitecture(Flowable):
    """Vector rendering of the Mermaid reference architecture in the source."""

    def __init__(self, width: float = 680, height: float = 320) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def _box(self, x, y, w, h, title, detail, fill, *, text=colors.white):
        c = self.canv
        c.setFillColor(fill)
        c.setStrokeColor(fill)
        c.roundRect(x, y, w, h, 7, stroke=1, fill=1)
        c.setFillColor(text)
        c.setFont(BOLD, 9)
        c.drawCentredString(x + w / 2, y + h / 2 + 4, title)
        if detail:
            c.setFont(FONT, 7.3)
            c.drawCentredString(x + w / 2, y + h / 2 - 9, detail)

    def _arrow(self, x1, y1, x2, y2):
        c = self.canv
        c.setStrokeColor(MUTED)
        c.setFillColor(MUTED)
        c.setLineWidth(1.5)
        c.line(x1, y1, x2, y2)
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 5
        for offset in (2.55, -2.55):
            c.line(x2, y2, x2 + size * math.cos(angle + offset), y2 + size * math.sin(angle + offset))

    def draw(self) -> None:
        c = self.canv
        self._box(8, 237, 100, 48, "Application teams", "many workloads", BLUE)
        self._box(138, 237, 126, 48, "Decision gateway", "one enterprise seam", NAVY)
        self._box(297, 237, 126, 48, "Routing policy", "quality | privacy | SLO | cost", TEAL)
        self._arrow(108, 261, 138, 261)
        self._arrow(264, 261, 297, 261)

        choices = [
            (5, "Trained encoder", "fixed, high-volume", BLUE),
            (174, "Shared-state encoder", "runtime labels", ORANGE),
            (343, "TypeSafe.ai JEV", "managed + flexible", PURPLE),
            (512, "Selective LLM", "complex / ambiguous", colors.HexColor("#334155")),
        ]
        for x, title, detail, fill in choices:
            self._box(x, 125, 153, 52, title, detail, fill)
            self._arrow(360, 237, x + 76, 177)

        self._box(217, 33, 246, 52, "Observed outcomes + labeled feedback", "evaluation, calibration, training", GREEN)
        for x, *_ in choices:
            self._arrow(x + 76, 125, 310, 85)
        c.setStrokeColor(GREEN)
        c.setLineWidth(1.6)
        c.line(463, 59, 630, 59)
        c.line(630, 59, 630, 213)
        c.line(630, 213, 423, 213)
        c.setFillColor(GREEN)
        c.setFont(BOLD, 8)
        c.drawString(485, 67, "calibration and routing updates")
        self._arrow(630, 213, 423, 237)


def page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont(FONT, 7.3)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, PAGE[1] - 24, "INSIDE CLASSIFIER TOKENOMICS | BLOG OUTLINE")
        canvas.drawRightString(PAGE[0] - doc.rightMargin, PAGE[1] - 24, "MARK AUSTIN")
        canvas.setStrokeColor(LINE)
        canvas.line(doc.leftMargin, PAGE[1] - 30, PAGE[0] - doc.rightMargin, PAGE[1] - 30)
    canvas.setStrokeColor(LINE)
    canvas.line(doc.leftMargin, 28, PAGE[0] - doc.rightMargin, 28)
    canvas.setFont(FONT, 7.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 15, "Independent classifier architecture, quality, cost, and deployment guide")
    canvas.drawRightString(PAGE[0] - doc.rightMargin, 15, str(doc.page))
    canvas.restoreState()


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def parse_markdown(md: str, s: dict[str, ParagraphStyle]) -> tuple[list, int]:
    lines = md.splitlines()
    story: list = []
    figure_count = 0
    i = 1  # the H1 title is handled on the cover
    paragraph_parts: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_parts:
            text = " ".join(item.strip() for item in paragraph_parts).strip()
            if text:
                story.append(Paragraph(markup(text), s["body"]))
            paragraph_parts.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            heading = stripped[3:]
            if heading.startswith(("4.", "7.", "8.", "10.")):
                story.append(PageBreak())
            story.append(Spacer(1, 6))
            story.append(Paragraph(markup(heading), s["h1"]))
            i += 1
            continue

        image_match = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            figure_count += 1
            alt, relative = image_match.groups()
            image_path = (SOURCE.parent / relative).resolve()
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            display_alt = {
                "Lowest modeled cost when a fixed trained taxonomy is eligible": "Fixed-taxonomy cost winners",
                "Lowest modeled cost when questions must change at runtime": "Runtime-question cost winners",
            }.get(alt, alt)
            caption = ""
            if i + 2 < len(lines) and not lines[i + 1].strip() and lines[i + 2].strip().startswith("*Figure"):
                caption = lines[i + 2].strip().strip("*")
                i += 2
            story.append(PageBreak())
            story.append(Paragraph(f"FIGURE {figure_count} | {markup(display_alt.upper())}", s["figure_label"]))
            image = scaled_image(image_path, 9.55 * inch, 5.55 * inch)
            story.append(Table([[image]], colWidths=[9.65 * inch], style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])) )
            if caption:
                story.append(Paragraph(markup(caption), s["caption"]))
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            if language == "mermaid":
                story.append(PageBreak())
                story.append(Paragraph("REFERENCE ARCHITECTURE", s["figure_label"]))
                story.append(ReferenceArchitecture())
                story.append(Paragraph("A hybrid decision gateway routes each workload according to quality, privacy, latency, flexibility, and cost, then feeds observed outcomes back into calibration and routing policy.", s["caption"]))
            else:
                code = "<br/>".join(markup(item) for item in block)
                story.append(Paragraph(code, s["code"]))
            i += 1
            continue

        bullet = re.match(r"^-\s+(.*)", stripped)
        if bullet:
            flush_paragraph()
            story.append(Paragraph(f"- {markup(bullet.group(1))}", s["bullet"]))
            i += 1
            continue

        numbered = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if numbered:
            flush_paragraph()
            story.append(Paragraph(f"{numbered.group(1)}. {markup(numbered.group(2))}", s["number"]))
            i += 1
            continue

        if stripped.startswith("*Figure"):
            # Captions are consumed with their images; this protects against a
            # malformed blank-line sequence without printing duplicate text.
            i += 1
            continue

        paragraph_parts.append(stripped)
        i += 1

    flush_paragraph()
    return story, figure_count


def build() -> None:
    s = styles()
    markdown = SOURCE.read_text()
    title = clean_ascii(markdown.splitlines()[0].removeprefix("# "))
    output_body, figure_count = parse_markdown(markdown, s)
    if figure_count != 12:
        raise ValueError(f"Expected 12 embedded charts, found {figure_count}")

    contents = [
        "1. One classification task, five execution patterns",
        "2. What JEV is - and what public evidence establishes",
        "3. The quality ladder from this benchmark",
        "4. Tokenomics: the variables that actually move cost",
        "5. Latency: resource time is not response time",
        "6. The data scientist's decision path",
        "7. The enterprise platform team's decision path",
        "8. When to move off a cloud API",
        "9. What we are testing now",
        "10. A pragmatic reference architecture",
        "11. Bottom line and visual editorial notes",
    ]

    story: list = [
        Spacer(1, 0.35 * inch),
        Paragraph("PRACTICAL GUIDE | PUBLICATION OUTLINE", s["kicker"]),
        Paragraph(markup(title), s["cover_title"]),
        Paragraph("How state length, question count, model quality, GPU utilization, and latency change the right classifier architecture", s["cover_sub"]),
        Spacer(1, 0.15 * inch),
        Table(
            [[Paragraph("12", s["callout"]), Paragraph("5", s["callout"]), Paragraph("2", s["callout"])],
             [Paragraph("embedded decision charts", s["small"]), Paragraph("variables that move the answer", s["small"]), Paragraph("operating perspectives", s["small"])]],
            colWidths=[2.15 * inch] * 3,
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]),
        ),
        Spacer(1, 0.27 * inch),
        Paragraph("For data scientists choosing a practical model and enterprise teams deciding when to operate a shared platform.", s["callout"]),
        Spacer(1, 0.15 * inch),
        Paragraph("Mark Austin | September 2026", s["cover_sub"]),
        PageBreak(),
        Paragraph("Guide structure", s["h1"]),
    ]
    for item in contents:
        story.append(Paragraph(markup(item), s["toc"]))
    story.extend([
        Spacer(1, 12),
        Paragraph("Visual logic", s["h2"]),
        Paragraph("Architecture comes before the leaderboard. Quality evidence comes before cost. State length and question count come before utilization. Deployment and operating-model choices come last, after the reader understands what each system actually computes.", s["body"]),
        PageBreak(),
    ])
    story.extend(output_body)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=PAGE,
        leftMargin=0.56 * inch,
        rightMargin=0.56 * inch,
        topMargin=0.52 * inch,
        bottomMargin=0.48 * inch,
        title=title,
        author="Mark Austin",
        subject="Practical guide to classifier tokenomics and model selection",
    )
    doc.build(story, onFirstPage=page_header_footer, onLaterPages=page_header_footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
