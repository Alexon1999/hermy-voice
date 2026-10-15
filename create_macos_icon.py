"""Génère l'icône .icns panda utilisée par Hermy.app."""
from __future__ import annotations

from pathlib import Path
import struct

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "hermy.icns"


def render_icon(size: int, path: Path) -> bytes:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 128, size / 128)
    gradient = QLinearGradient(0, 0, 128, 128)
    gradient.setColorAt(0, QColor("#8f7cff"))
    gradient.setColorAt(1, QColor("#46c8bd"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    painter.drawRoundedRect(QRectF(4, 4, 120, 120), 29, 29)
    dark = QColor("#24242b")
    painter.setBrush(dark)
    painter.drawEllipse(QRectF(23, 18, 38, 40))
    painter.drawEllipse(QRectF(67, 18, 38, 40))
    painter.setBrush(QColor("#fffaf1"))
    painter.drawEllipse(QRectF(19, 31, 90, 83))
    painter.setBrush(dark)
    painter.drawEllipse(QRectF(31, 51, 31, 36))
    painter.drawEllipse(QRectF(66, 51, 31, 36))
    painter.setBrush(QColor("#fffaf1"))
    painter.drawEllipse(QRectF(42, 61, 10, 13))
    painter.drawEllipse(QRectF(77, 61, 10, 13))
    painter.setBrush(dark)
    painter.drawEllipse(QRectF(46, 65, 5, 7))
    painter.drawEllipse(QRectF(81, 65, 5, 7))
    painter.drawEllipse(QRectF(58, 82, 13, 9))
    painter.setPen(QPen(dark, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    smile = QPainterPath()
    smile.moveTo(64, 89)
    smile.cubicTo(58, 100, 49, 98, 47, 94)
    smile.moveTo(64, 89)
    smile.cubicTo(70, 100, 79, 98, 81, 94)
    painter.drawPath(smile)
    painter.end()
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"Impossible d'écrire l'image : {path}")
    return path.read_bytes()


def main() -> None:
    icon_dir = ROOT / "build" / "macos"
    icon_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for chunk_type, size in (
        (b"icp4", 16), (b"icp5", 32), (b"icp6", 64),
        (b"ic07", 128), (b"ic08", 256), (b"ic09", 512),
        (b"ic10", 1024),
    ):
        png = render_icon(size, icon_dir / f"hermy_{size}.png")
        chunks.append(struct.pack(">4sI", chunk_type, len(png) + 8) + png)
    body = b"".join(chunks)
    OUTPUT.write_bytes(struct.pack(">4sI", b"icns", len(body) + 8) + body)
    print(f"Icône Hermy créée : {OUTPUT}")


if __name__ == "__main__":
    main()
