import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/concentration_agreement_evaluation.xlsx";
const previewDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/.codex_work/spreadsheet_inspect/outputs_agreement_preview";
await fs.mkdir(previewDir, { recursive: true });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
for (const query of [
  { kind: "sheet", include: "id,name", maxChars: 5000 },
  { kind: "table", sheetId: "Processed Series", range: "A1:T30", include: "values,formulas", tableMaxRows: 30, tableMaxCols: 20, maxChars: 15000 },
  { kind: "drawing", sheetId: "Processed Series", maxChars: 8000 },
  { kind: "table", sheetId: "Dashboard", range: "A1:Q19", include: "values,formulas", tableMaxRows: 19, tableMaxCols: 17, maxChars: 18000 },
  { kind: "drawing", sheetId: "Dashboard", maxChars: 8000 },
  { kind: "table", sheetId: "Evaluation", range: "A3:F13", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 6, maxChars: 10000 },
]) process.stdout.write((await workbook.inspect(query)).ndjson + "\n");
for (const [name, sheetName, range] of [
  ["processed", "Processed Series", "A1:T30"],
  ["dashboard", "Dashboard", "A1:T43"],
]) {
  const image = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await image.arrayBuffer()));
}
