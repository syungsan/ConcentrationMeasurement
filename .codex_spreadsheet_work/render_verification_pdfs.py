from pathlib import Path
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


root = Path(__file__).resolve().parents[1]
pdf_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "outputs" / "fixed_concentration_agreement" / "verification_pdfs"
png_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / ".codex_spreadsheet_work" / "excel_renders"
png_dir.mkdir(parents=True, exist_ok=True)
renders = []
for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    doc = pdfium.PdfDocument(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        bitmap = page.render(scale=1.4)
        out = png_dir / f"{pdf_path.stem}_{page_index + 1}.png"
        bitmap.to_pil().convert("RGB").save(out)
        image = Image.open(out).convert("RGB")
        image.thumbnail((700, 500))
        renders.append((f"{pdf_path.stem} p{page_index + 1}", image.copy()))
        page.close()
    doc.close()

width = 1500
tile_h = 570
rows = (len(renders) + 1) // 2
sheet = Image.new("RGB", (width, rows * tile_h), "white")
draw = ImageDraw.Draw(sheet)
for index, (name, image) in enumerate(renders):
    x = (index % 2) * 750
    y = (index // 2) * tile_h
    draw.text((x + 12, y + 10), name, fill="black")
    sheet.paste(image, (x + 12, y + 40))
sheet.save(png_dir / "all_sheets.png")
print(png_dir / "all_sheets.png")
