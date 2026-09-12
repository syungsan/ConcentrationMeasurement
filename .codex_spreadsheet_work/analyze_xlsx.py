from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def analyze(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pr:Relationship", NS)
        }
        out = {"file": str(path), "sheets": [], "defined_names": []}
        defined = wb.find("m:definedNames", NS)
        if defined is not None:
            for item in defined:
                out["defined_names"].append({
                    "name": item.attrib.get("name"),
                    "localSheetId": item.attrib.get("localSheetId"),
                    "formula": item.text,
                })
        for sheet in wb.find("m:sheets", NS):
            name = sheet.attrib["name"]
            target = relmap[sheet.attrib[f"{{{NS['r']}}}id"]]
            xml_path = target.lstrip("/")
            if not xml_path.startswith("xl/"):
                xml_path = "xl/" + xml_path
            xml_path = re.sub(r"xl/worksheets/\.\./", "xl/", xml_path)
            root = ET.fromstring(zf.read(xml_path))
            dim = root.find("m:dimension", NS)
            formulas = []
            errors = []
            types = Counter()
            for cell in root.findall(".//m:c", NS):
                types[cell.attrib.get("t", "n")] += 1
                f = cell.find("m:f", NS)
                v = cell.find("m:v", NS)
                if f is not None:
                    formulas.append({"cell": cell.attrib.get("r"), "formula": f.text or "", "cached": v.text if v is not None else None})
                if cell.attrib.get("t") == "e":
                    errors.append({"cell": cell.attrib.get("r"), "value": v.text if v is not None else None})
            drawing = root.find("m:drawing", NS)
            tables = root.find("m:tableParts", NS)
            out["sheets"].append({
                "name": name,
                "path": xml_path,
                "dimension": dim.attrib.get("ref") if dim is not None else None,
                "cell_types": dict(types),
                "formula_count": len(formulas),
                "formula_samples": formulas[:20] + formulas[-20:] if len(formulas) > 40 else formulas,
                "errors": errors,
                "has_drawing": drawing is not None,
                "table_parts": int(tables.attrib.get("count", "0")) if tables is not None else 0,
            })
        out["table_xml"] = {}
        for name in zf.namelist():
            if name.startswith("xl/tables/") and name.endswith(".xml"):
                table = ET.fromstring(zf.read(name))
                out["table_xml"][name] = {
                    "name": table.attrib.get("name"),
                    "displayName": table.attrib.get("displayName"),
                    "ref": table.attrib.get("ref"),
                    "autoFilter": (table.find("m:autoFilter", NS).attrib.get("ref") if table.find("m:autoFilter", NS) is not None else None),
                }
        out["chart_xml"] = sorted(n for n in zf.namelist() if n.startswith("xl/charts/chart") and n.endswith(".xml"))
        return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for value in sys.argv[1:]:
        result = analyze(Path(value))
        print(json.dumps(result, ensure_ascii=False, indent=2))
