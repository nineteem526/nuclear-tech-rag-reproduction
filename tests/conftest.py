from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas


@pytest.fixture
def structured_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "reactor-cooling-system.pdf"
    canvas = Canvas(str(path), pagesize=A4)

    canvas.setFont("Helvetica-Bold", 20)
    canvas.drawString(72, 790, "1 Reactor Cooling System")
    canvas.setFont("Helvetica", 11)
    text = canvas.beginText(72, 750)
    text.textLine("The reactor coolant pumps circulate primary coolant.")
    text.textLine("Each train has an independent monitoring channel.")
    canvas.drawText(text)
    canvas.showPage()

    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(72, 790, "1.1 Pump Requirements")
    canvas.setFont("Helvetica", 11)
    text = canvas.beginText(72, 750)
    text.textLine("Pump RCP-001 shall remain available during normal operation.")
    text.textLine("Alarm code ALM-204 indicates abnormal bearing temperature.")
    canvas.drawText(text)
    canvas.save()
    return path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "empty.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    canvas.showPage()
    canvas.save()
    return path

