import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const source = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/predictions/20260714_unnan_nishi_6-1_2_full/analysis/concentration_prediction_agreement_analysis.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const raw = workbook.worksheets.getItem("Raw Data");
const values = raw.getRange("A1:J3315").values;
const secCounts = new Map();
let maxSec = -Infinity;
let standards = 0;
for (const row of values.slice(1)) {
  const sec = Number(row[1]);
  if (Number.isFinite(sec)) {
    maxSec = Math.max(maxSec, sec);
    secCounts.set(sec, (secCounts.get(sec) || 0) + 1);
  }
  if (row[6] !== null && row[6] !== "") standards += 1;
}
const duplicates = [...secCounts.entries()].filter(([, count]) => count > 1);
process.stdout.write(JSON.stringify({ rows: values.length - 1, maxSec, uniqueSeconds: secCounts.size, duplicateSeconds: duplicates.length, maxDuplicateCount: Math.max(...secCounts.values()), standards, tail: values.slice(-5) }, null, 2));
