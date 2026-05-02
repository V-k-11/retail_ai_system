from __future__ import annotations

import re
from pathlib import Path
from textwrap import wrap


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "project_explanation.md"
TARGET = ROOT / "docs" / "retail_ai_project_explanation.pdf"


def clean_line(line: str) -> str:
    line = line.replace("↓", "->")
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = line.replace("|", " ")
    line = line.replace("—", "-")
    line = line.replace("–", "-")
    return line.encode("latin-1", "replace").decode("latin-1")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def source_lines() -> list[str]:
    lines = []
    in_code = False
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not line:
            lines.append("")
            continue
        if line.startswith("# "):
            lines.append(clean_line(line[2:].upper()))
            lines.append("")
        elif line.startswith("## "):
            lines.append(clean_line(line[3:]))
            lines.append("")
        elif line.startswith("### "):
            lines.append(clean_line(line[4:]))
        elif line.startswith("- "):
            lines.extend(wrap("- " + clean_line(line[2:]), width=92))
        elif in_code:
            lines.append("  " + clean_line(line))
        else:
            lines.extend(wrap(clean_line(line), width=96))
    return lines


def paginate(lines: list[str], per_page: int = 48) -> list[list[str]]:
    pages = []
    current = []
    for line in lines:
        current.append(line)
        if len(current) >= per_page:
            pages.append(current)
            current = []
    if current:
        pages.append(current)
    return pages


def build_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []

    def add_object(body: str | bytes) -> int:
        if isinstance(body, str):
            body = body.encode("latin-1")
        objects.append(body)
        return len(objects)

    catalog_id = add_object("")
    pages_id = add_object("")
    font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids = []
    for page in pages:
        commands = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line in page:
            commands.append(f"({pdf_escape(line)}) Tj")
            commands.append("T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        content_id = add_object(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_id = add_object(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 842] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)

    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    pages = paginate(source_lines())
    TARGET.write_bytes(build_pdf(pages))
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()

