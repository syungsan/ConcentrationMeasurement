import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/concentration_agreement_evaluation.xlsx";
const previewDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/.codex_work/spreadsheet_inspect/outputs_agreement_fixed";
await fs.mkdir(previewDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const processed = workbook.worksheets.getItem("Processed Series");
const dashboard = workbook.worksheets.getItem("Dashboard");

// Use only populated seconds and moving-average values; template rows are excluded.
const processedValues = processed.getUsedRange().values;
let lastProcessedRow = 4;
const uniqueSeconds = new Set();
for (let index = 3; index < processedValues.length; index += 1) {
  if (typeof processedValues[index]?.[0] === "number" && typeof processedValues[index]?.[2] === "number") {
    lastProcessedRow = index + 1;
    uniqueSeconds.add(processedValues[index][0]);
  }
}
const sortedSeconds = [...uniqueSeconds].sort((a, b) => a - b);
const helperFirstRow = 4;
const helperLastRow = helperFirstRow + sortedSeconds.length - 1;

processed.getRange("V3:W4000").clear({ applyTo: "all" });
processed.getRange("V3:W3").values = [["chart_sec", "chart_score_mean_display_MA"]];
processed.getRange("V3:W3").copyFrom(processed.getRange("A3:B3"), "formats");
processed.getRange(`V${helperFirstRow}:V${helperLastRow}`).values = sortedSeconds.map((sec) => [sec]);
processed.getRange(`W${helperFirstRow}:W${helperLastRow}`).formulas = sortedSeconds.map((_, index) => [
  `=AVERAGEIF($A$4:$A$${lastProcessedRow},V${helperFirstRow + index},$C$4:$C$${lastProcessedRow})`,
]);
processed.getRange(`V${helperFirstRow}:V${helperLastRow}`).format.numberFormat = "0";
processed.getRange(`W${helperFirstRow}:W${helperLastRow}`).format.numberFormat = "0.0000";
processed.getRange("V:W").format.columnWidth = 24;

// Rebuild the overall moving-average chart as one opaque series, without legend/note.
processed.charts.deleteAll();
processed.getRange("H27:T27").unmerge();
processed.getRange("H27:T27").clear({ applyTo: "all" });
const overallChart = processed.charts.add("line", {
  title: "全期間の集中度推移（移動平均）",
  hasLegend: false,
});
const overallSeries = overallChart.series.add("score_mean_display_MA");
overallSeries.categoryFormula = `'Processed Series'!$V$${helperFirstRow}:$V$${helperLastRow}`;
overallSeries.formula = `'Processed Series'!$W$${helperFirstRow}:$W$${helperLastRow}`;
overallSeries.fill = "#1F6D8C";
overallChart.hasLegend = false;
overallChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
overallChart.yAxis = { min: 1, max: 7, numberFormatCode: "0.0", textStyle: { fontSize: 9 } };
overallChart.setPosition("H2", "T25");

// Rebuild Dashboard charts. The AI comparison now uses the ±5-second local mean
// already calculated from Raw Data in Dashboard column B / Evaluation column C.
dashboard.charts.deleteAll();
const trendChart = dashboard.charts.add("line", {
  title: "AI局所平均（±5秒）と人手評価",
  hasLegend: true,
});
for (const [name, col, color] of [
  ["AI local mean (±5 sec)", "B", "#1F6D8C"],
  ["gold", "C", "#ED7D31"],
  ["bronze", "D", "#70AD47"],
  ["tadano", "E", "#5B9BD5"],
]) {
  const series = trendChart.series.add(name);
  series.categoryFormula = `'Dashboard'!$A$10:$A$19`;
  series.formula = `'Dashboard'!$${col}$10:$${col}$19`;
  series.fill = color;
}
trendChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
trendChart.yAxis = { min: 1, max: 7, numberFormatCode: "0.0", textStyle: { fontSize: 9 } };
trendChart.setPosition("A22", "J42");

const changeChart = dashboard.charts.add("line", dashboard.getRange("H9:L19"));
changeChart.title = "Change Between Human-Rating Times";
changeChart.hasLegend = true;
changeChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
changeChart.yAxis = { min: -3, max: 3, numberFormatCode: "0.0", textStyle: { fontSize: 9 } };
changeChart.setPosition("K22", "T42");

for (const [name, sheetName, range] of [
  ["processed", "Processed Series", "A1:T29"],
  ["dashboard", "Dashboard", "A1:T43"],
]) {
  const image = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await image.arrayBuffer()));
}

for (const query of [
  { kind: "table", sheetId: "Dashboard", range: "A9:F19", include: "values,formulas", tableMaxRows: 11, tableMaxCols: 6, maxChars: 10000 },
  { kind: "drawing", sheetId: "Processed Series", maxChars: 5000 },
  { kind: "drawing", sheetId: "Dashboard", maxChars: 8000 },
]) process.stdout.write((await workbook.inspect(query)).ndjson + "\n");

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "final formula error scan",
});
process.stdout.write(errors.ndjson + "\n");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path);
process.stdout.write(JSON.stringify({ saved: path, lastProcessedRow, helperLastRow }) + "\n");
