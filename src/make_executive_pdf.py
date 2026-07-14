"""
make_executive_pdf.py
----------------------
Builds a polished, client-ready PDF version of reports/executive_summary.md
using reportlab. Run:
    python src/make_executive_pdf.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "executive_summary.pdf"

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#1B2A4A"))
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#555555"), spaceAfter=4)
h2_style = ParagraphStyle("H2Custom", parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#1B2A4A"), spaceBefore=16, spaceAfter=8)
body_style = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8)

story = []

story.append(Paragraph("Executive Summary", title_style))
story.append(Paragraph("Customer Segmentation Analysis", ParagraphStyle("st2", parent=subtitle_style, fontSize=14, textColor=colors.HexColor("#333333"))))
story.append(Spacer(1, 6))
story.append(Paragraph("Prepared for: Marketing &amp; Growth Leadership", subtitle_style))
story.append(Paragraph("Prepared by: Data Analytics Team", subtitle_style))
story.append(Paragraph("Date: July 2026", subtitle_style))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", color=colors.HexColor("#1B2A4A"), thickness=1.2))
story.append(Spacer(1, 14))

story.append(Paragraph("Objective", h2_style))
story.append(Paragraph(
    "Identify distinct customer groups within the retail customer base so marketing spend can be "
    "targeted rather than blanket, improving retention and revenue efficiency.", body_style))

story.append(Paragraph("Approach", h2_style))
story.append(Paragraph(
    "We analyzed 1,200 customers (built on a real Kaggle customer dataset, expanded with modeled "
    "behavioral data) across seven behavioral dimensions: age, income, spending score, purchase "
    "frequency, recency, tenure, and monetary value. Three clustering algorithms &mdash; K-Means, "
    "Hierarchical Clustering, and DBSCAN &mdash; were tested and compared. K-Means (k=4) was "
    "selected as the production model based on cluster balance, interpretability, and stability "
    "testing.", body_style))

story.append(Paragraph("The Four Segments", h2_style))

table_data = [
    ["Segment", "Share", "Profile", "Recommended Action"],
    ["Premium\nSpenders", "21%", "Highest income & spending,\nmost frequent, most recent,\nlongest tenure",
     "VIP loyalty tier, priority\nsupport, referral program"],
    ["Steady Mature\nRegulars", "24%", "Older, mid income/spend,\ndependable",
     "Cross-sell campaigns,\nmid-tier loyalty perks"],
    ["New Bargain\nHunters", "28%", "Younger, price-sensitive,\nengaged, growing",
     "Time-boxed discounts,\nbundle deals, rewards"],
    ["At-Risk\nCustomers", "27%", "High income but lowest\nengagement, ~120 days\nsince last order",
     "Urgent win-back campaign,\nroot-cause survey"],
]

t = Table(table_data, colWidths=[1.15 * inch, 0.6 * inch, 2.15 * inch, 2.1 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B2A4A")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
]))
story.append(t)
story.append(Spacer(1, 14))

story.append(Paragraph("Key Findings", h2_style))
findings = [
    "<b>No single metric predicts value.</b> Income alone does not predict spending &mdash; multi-dimensional clustering was necessary to find real structure.",
    "<b>At-Risk Customers are the highest-leverage opportunity.</b> They have the income to spend but have quietly disengaged &mdash; unlike low-income segments, the barrier isn't affordability, making them a strong win-back target.",
    "<b>The segmentation is statistically stable.</b> Repeated testing on random subsamples showed consistent results, meaning these segments can be trusted for ongoing campaign planning, not just a one-time snapshot.",
    "<b>DBSCAN was evaluated but rejected for production use</b> &mdash; while it scored well numerically, it produced one oversized cluster and many tiny, unusable fragments rather than clean, equal-sized business segments.",
]
for f in findings:
    story.append(Paragraph(f"&bull; {f}", body_style))

story.append(Paragraph("Recommended Next Steps", h2_style))
steps = [
    "Launch a win-back campaign for At-Risk Customers within the next marketing cycle.",
    "Pilot a VIP loyalty tier for Premium Spenders to protect and grow this high-value segment.",
    "Re-run this segmentation quarterly and track segment migration over time.",
    "Replace engineered behavioral fields with real transaction data as it becomes available.",
]
for i, s in enumerate(steps, 1):
    story.append(Paragraph(f"{i}. {s}", body_style))

story.append(Paragraph("Expected Business Impact", h2_style))
story.append(Paragraph(
    "Targeted campaigns per segment (instead of one blanket strategy) are expected to improve "
    "campaign response rates and reduce wasted spend on customers unlikely to respond to generic "
    "offers, while directly addressing the revenue at risk in the At-Risk segment.", body_style))

story.append(Spacer(1, 16))
story.append(HRFlowable(width="100%", color=colors.HexColor("#CCCCCC"), thickness=0.7))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<i>Full technical methodology, code, and visualizations are available in the accompanying "
    "Jupyter notebook and GitHub repository.</i>",
    ParagraphStyle("footer", parent=body_style, fontSize=9, textColor=colors.HexColor("#777777"))
))

doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                         topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                         leftMargin=0.75 * inch, rightMargin=0.75 * inch)
doc.build(story)
print(f"Saved: {OUT}")
