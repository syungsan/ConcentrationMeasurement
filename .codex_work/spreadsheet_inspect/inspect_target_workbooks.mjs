import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const base = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/predictions/20260714_unnan_nishi_6-1_2_full";
const previewDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/.codex_work/spreadsheet_inspect/target_previews";
await fs.mkdir(previewDir, { recursive: true });

for (const name of ["concentration_agreement_evaluation.xlsx", "concentration_agreement_evaluation2.xlsx"]) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(`${base}/${name}`));
  const stem = name.replace(".xlsx", "");
  process.stdout.write(`${name}\n`);
  process.stdout.write((await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 5000 })).ndjson + "\n");
  process.stdout.write((await workbook.inspect({ kind: "table", sheetId: "Processed Series", range: "A1:H20", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 8, maxChars: 12000 })).ndjson + "\n");
  process.stdout.write((await workbook.inspect({ kind: "drawing", sheetId: "Processed Series", maxChars: 5000 })).ndjson + "\n");
  const image = await workbook.render({ sheetName: "Processed Series", range: "A1:N35", scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${stem}.png`, new Uint8Array(await image.arrayBuffer()));
}
