import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "../outputs/20260714_unnan_nishi_6-1_2_full/019f8ee8-9b86-7b50-881b-60f28d53a92b/concentration_prediction_agreement_analysis.xlsx";
const outputDir = "./agreement_verify";
await fs.mkdir(outputDir, { recursive: true });

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const summary = await workbook.inspect({
  kind: "sheet,table,drawing",
  maxChars: 6000,
  tableMaxRows: 8,
  tableMaxCols: 8,
});
console.log(summary.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

for (const [sheetName, range] of [
  ["Time Series Chart", "A1:Q28"],
  ["Agreement Analysis", "A1:AS43"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const bytes = new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(`${outputDir}/${sheetName.replaceAll(" ", "_")}.png`, bytes);
}
