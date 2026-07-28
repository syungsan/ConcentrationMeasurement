import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/20260714_unnan_nishi_6-1_2_full/019f8ee8-9b86-7b50-881b-60f28d53a92b/concentration_prediction_agreement_analysis.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));

for (const query of [
  { kind: "sheet", include: "id,name", maxChars: 5000 },
  { kind: "table", maxChars: 12000, tableMaxRows: 15, tableMaxCols: 12 },
  { kind: "formula", maxChars: 16000, options: { maxResults: 250 } },
  { kind: "drawing", maxChars: 12000 },
]) {
  const result = await workbook.inspect(query);
  process.stdout.write(result.ndjson + "\n");
}
