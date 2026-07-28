import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/predictions/20260313_unnan_nishi_5-1_1_full/ab_intervention_report.xlsx";
const previewDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/.codex_work/spreadsheet_inspect/ab_report_final_previews";
await fs.mkdir(previewDir, { recursive: true });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 10000 });
process.stdout.write(sheets.ndjson + "\n");
for (const [name, range] of [
  ["Dashboard", "A1:W72"],
  ["README", "A1:H14"],
  ["Events", "A1:J12"],
  ["Time Course", "A1:Q28"],
  ["Situation Summary", "A1:I20"],
  ["Effects", "A1:L12"],
  ["Cooldown Events", "A1:M20"],
  ["Cooldown Summary", "A1:F20"],
]) {
  const image = await workbook.render({ sheetName: name, range, scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name.replaceAll(" ", "_")}.png`, new Uint8Array(await image.arrayBuffer()));
}
for (const query of [
  { kind: "table", sheetId: "Effects", range: "A3:L10", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 12, maxChars: 10000 },
  { kind: "table", sheetId: "Cooldown Summary", range: "A3:F20", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 6, maxChars: 10000 },
  { kind: "drawing", sheetId: "Dashboard", maxChars: 10000 },
]) process.stdout.write((await workbook.inspect(query)).ndjson + "\n");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "AB report formula error scan",
});
process.stdout.write(errors.ndjson + "\n");
