import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/20260714_unnan_nishi_6-1_2_full/019f8ee8-9b86-7b50-881b-60f28d53a92b/concentration_prediction_agreement_analysis.xlsx";
const outputDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/019f938b-55cc-72c2-b702-30610ee27958";
const outputPath = `${outputDir}/concentration_prediction_agreement_analysis_pm5s.xlsx`;
const previewDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/.codex_work/spreadsheet_inspect/previews_pm5s";

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));

async function saveRender(name, options) {
  const image = await workbook.render({ ...options, scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await image.arrayBuffer()));
}

// Baseline visual inspection before making targeted changes.
await saveRender("before_raw", { sheetName: "Raw Data", range: "A1:J30" });
await saveRender("before_time_series", { sheetName: "Time Series Chart", range: "A1:N30" });
await saveRender("before_agreement", { sheetName: "Agreement Analysis", range: "A1:N43" });

const analysis = workbook.worksheets.getItem("Agreement Analysis");
analysis.getRange("A1").values = [["Prediction Accuracy & Agreement Analysis (±5-sec local mean)"]];
analysis.getRange("A3:D3").values = [[
  "evaluation_time (sec)",
  "window_start (sec)",
  "observed_seconds",
  "score_mean_local_pm5s",
]];

const evaluationTimes = Array.from({ length: 10 }, (_, index) => index * 300);
analysis.getRange("A4:A13").values = evaluationTimes.map((time) => [time]);
analysis.getRange("B4:B13").formulas = evaluationTimes.map((_, index) => [
  `=MAX(0,A${index + 4}-5)`,
]);
analysis.getRange("C4:C13").formulas = evaluationTimes.map((_, index) => [
  `=COUNTIFS('Time Series Chart'!$A$4:$A$2711,\">=\"&MAX(0,A${index + 4}-5),'Time Series Chart'!$A$4:$A$2711,\"<=\"&A${index + 4}+5)`,
]);
analysis.getRange("D4:D13").formulas = evaluationTimes.map((_, index) => [
  `=AVERAGEIFS('Time Series Chart'!$B$4:$B$2711,'Time Series Chart'!$A$4:$A$2711,\">=\"&MAX(0,A${index + 4}-5),'Time Series Chart'!$A$4:$A$2711,\"<=\"&A${index + 4}+5)`,
]);
analysis.getRange("B34").formulas = [[
  '=\"score_mean uses an inclusive ±5-sec local mean at each human-rating time (up to 11 one-second observations; 0 sec uses 0-5 sec).\"',
]];
analysis.getRange("A33").values = [["Minimum observed seconds/window"]];

const comparisonChart = analysis.charts.items[0];
comparisonChart.title = "±5-sec Local Mean Four-Series Comparison";
comparisonChart.series.items[0].name = "score_mean_local_pm5s";

// Keep the existing calculation layout and chart, which already reference D4:G13.
await saveRender("after_raw", { sheetName: "Raw Data", range: "A1:J30" });
await saveRender("after_time_series", { sheetName: "Time Series Chart", range: "A1:N30" });
await saveRender("after_agreement", { sheetName: "Agreement Analysis", range: "A1:N43" });

const keyTable = await workbook.inspect({
  kind: "table",
  sheetId: "Agreement Analysis",
  range: "A1:G43",
  include: "values,formulas",
  tableMaxRows: 43,
  tableMaxCols: 7,
  maxChars: 20000,
});
process.stdout.write(keyTable.ndjson + "\n");

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
process.stdout.write(formulaErrors.ndjson + "\n");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
process.stdout.write(JSON.stringify({ outputPath }) + "\n");
