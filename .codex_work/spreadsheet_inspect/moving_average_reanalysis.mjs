import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/20260714_unnan_nishi_6-1_2_full/019f8ee8-9b86-7b50-881b-60f28d53a92b/concentration_prediction_agreement_analysis.xlsx";
const outputDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/019f938b-55cc-72c2-b702-30610ee27958";
const outputPath = `${outputDir}/concentration_prediction_moving_average_sensitivity.xlsx`;
const previewDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/.codex_work/spreadsheet_inspect/previews_moving_average";
const windows = [5, 11, 21, 31, 61];
const halfWidths = [2, 5, 10, 15, 30];
const lastSecond = 2707;

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));

async function renderTo(name, options) {
  const image = await workbook.render({ ...options, scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await image.arrayBuffer()));
}

// Visual baseline for every source sheet before modification.
await renderTo("before_raw", { sheetName: "Raw Data", range: "A1:J30" });
await renderTo("before_time_series", { sheetName: "Time Series Chart", range: "A1:N30" });
await renderTo("before_agreement", { sheetName: "Agreement Analysis", range: "A1:N43" });

const moving = workbook.worksheets.add("Moving Average Series");
moving.showGridLines = false;
moving.getRange("A1:P1").merge();
moving.getRange("A1").values = [["Centered Moving Averages of score_mean"]];
moving.getRange("A3:J3").values = [[
  "time (sec)", "score_mean_raw", "MA_5s", "MA_11s", "MA_21s", "MA_31s", "MA_61s",
  "gold_standard", "bronze_standard", "tadano",
]];

const timeFormulas = [];
const rawFormulas = [];
const maFormulas = windows.map(() => []);
const standardFormulas = [[], [], []];
for (let sec = 0; sec <= lastSecond; sec += 1) {
  const sourceRow = sec + 4;
  timeFormulas.push([`='Time Series Chart'!A${sourceRow}`]);
  rawFormulas.push([`='Time Series Chart'!B${sourceRow}`]);
  for (let w = 0; w < windows.length; w += 1) {
    const startRow = 4 + Math.max(0, sec - halfWidths[w]);
    const endRow = 4 + Math.min(lastSecond, sec + halfWidths[w]);
    maFormulas[w].push([`=AVERAGE($B$${startRow}:$B$${endRow})`]);
  }
  for (let s = 0; s < 3; s += 1) {
    standardFormulas[s].push([`='Time Series Chart'!${String.fromCharCode(67 + s)}${sourceRow}`]);
  }
}
moving.getRange("A4:A2711").formulas = timeFormulas;
moving.getRange("B4:B2711").formulas = rawFormulas;
for (let w = 0; w < windows.length; w += 1) {
  const col = String.fromCharCode(67 + w);
  moving.getRange(`${col}4:${col}2711`).formulas = maFormulas[w];
}
for (let s = 0; s < 3; s += 1) {
  const col = String.fromCharCode(72 + s);
  moving.getRange(`${col}4:${col}2711`).formulas = standardFormulas[s];
}

moving.getRange("K3:P3").values = [[
  "time (sec)", "score_mean_raw", "MA_11s", "gold_standard", "bronze_standard", "tadano",
]];
moving.getRange("K4:P2711").formulas = Array.from({ length: 2708 }, (_, i) => {
  const row = i + 4;
  return [`=A${row}`, `=B${row}`, `=D${row}`, `=H${row}`, `=I${row}`, `=J${row}`];
});

const trendChart = moving.charts.add("line", moving.getRange("K3:P2711"));
trendChart.title = "Raw score_mean, 11-sec Moving Average, and Human Ratings";
trendChart.hasLegend = true;
trendChart.yAxis = { min: 1, max: 7, numberFormatCode: "0.0" };
trendChart.xAxis = { axisType: "textAxis" };
trendChart.setPosition("R2", "AB25");

const sensitivity = workbook.worksheets.add("Window Sensitivity");
sensitivity.showGridLines = false;
sensitivity.getRange("A1:V1").merge();
sensitivity.getRange("A1").values = [["Moving-Average Window Sensitivity at 300-sec Human-Rating Times"]];
sensitivity.getRange("A3:N3").values = [[
  "evaluation_time (sec)", "MA_5s", "MA_11s", "MA_21s", "MA_31s", "MA_61s",
  "gold_standard", "bronze_standard", "tadano",
  "n_5s", "n_11s", "n_21s", "n_31s", "n_61s",
]];
sensitivity.getRange("O3:V3").values = [[
  "rank_MA_5s", "rank_MA_11s", "rank_MA_21s", "rank_MA_31s", "rank_MA_61s",
  "rank_gold", "rank_bronze", "rank_tadano",
]];

const evaluationTimes = Array.from({ length: 10 }, (_, index) => index * 300);
sensitivity.getRange("A4:A13").values = evaluationTimes.map((time) => [time]);
for (let w = 0; w < windows.length; w += 1) {
  const sampleCol = String.fromCharCode(66 + w);
  const sourceCol = String.fromCharCode(67 + w);
  sensitivity.getRange(`${sampleCol}4:${sampleCol}13`).formulas = evaluationTimes.map((_, i) => {
    const row = i + 4;
    return [`=INDEX('Moving Average Series'!$${sourceCol}$4:$${sourceCol}$2711,MATCH($A${row},'Moving Average Series'!$A$4:$A$2711,0))`];
  });
  const countCol = String.fromCharCode(74 + w);
  sensitivity.getRange(`${countCol}4:${countCol}13`).formulas = evaluationTimes.map((_, i) => {
    const row = i + 4;
    return [`=COUNTIFS('Moving Average Series'!$A$4:$A$2711,\">=\"&MAX(0,$A${row}-${halfWidths[w]}),'Moving Average Series'!$A$4:$A$2711,\"<=\"&MIN(${lastSecond},$A${row}+${halfWidths[w]}))`];
  });
  const rankCol = String.fromCharCode(79 + w);
  sensitivity.getRange(`${rankCol}4:${rankCol}13`).formulas = evaluationTimes.map((_, i) => {
    const row = i + 4;
    return [`=RANK(${sampleCol}${row},$${sampleCol}$4:$${sampleCol}$13,1)`];
  });
}
for (let s = 0; s < 3; s += 1) {
  const sampleCol = String.fromCharCode(71 + s);
  const sourceCol = String.fromCharCode(72 + s);
  sensitivity.getRange(`${sampleCol}4:${sampleCol}13`).formulas = evaluationTimes.map((_, i) => {
    const row = i + 4;
    return [`=INDEX('Moving Average Series'!$${sourceCol}$4:$${sourceCol}$2711,MATCH($A${row},'Moving Average Series'!$A$4:$A$2711,0))`];
  });
  const rankCol = String.fromCharCode(84 + s);
  sensitivity.getRange(`${rankCol}4:${rankCol}13`).formulas = evaluationTimes.map((_, i) => {
    const row = i + 4;
    return [`=RANK(${sampleCol}${row},$${sampleCol}$4:$${sampleCol}$13,1)`];
  });
}

sensitivity.getRange("A15:J15").values = [[
  "window_seconds", "comparison", "N", "MAE", "RMSE", "Bias", "Pearson r", "Spearman rho", "CCC", "R^2",
]];
const comparisonNames = ["score vs gold", "score vs bronze", "score vs tadano"];
const refCols = ["G", "H", "I"];
const refRankCols = ["T", "U", "V"];
const metricRows = [];
const absHelperHeaders = [];
const absHelperFormulas = [];
let metricRow = 16;
let helperIndex = 0;
for (let w = 0; w < windows.length; w += 1) {
  const scoreCol = String.fromCharCode(66 + w);
  const scoreRankCol = String.fromCharCode(79 + w);
  for (let s = 0; s < 3; s += 1) {
    const refCol = refCols[s];
    const helperColNumber = 24 + helperIndex;
    const helperCol = helperColNumber <= 26
      ? String.fromCharCode(64 + helperColNumber)
      : `A${String.fromCharCode(64 + helperColNumber - 26)}`;
    absHelperHeaders.push(`${windows[w]}s_${comparisonNames[s]}_abs_error`);
    absHelperFormulas.push(evaluationTimes.map((_, i) => {
      const row = i + 4;
      return [`=ABS(${scoreCol}${row}-${refCol}${row})`];
    }));
    metricRows.push({ metricRow, w, s, scoreCol, scoreRankCol, refCol, refRankCol: refRankCols[s], helperCol });
    metricRow += 1;
    helperIndex += 1;
  }
}

for (let h = 0; h < absHelperHeaders.length; h += 1) {
  const colNumber = 24 + h;
  const col = colNumber <= 26
    ? String.fromCharCode(64 + colNumber)
    : `A${String.fromCharCode(64 + colNumber - 26)}`;
  sensitivity.getRange(`${col}3`).values = [[absHelperHeaders[h]]];
  sensitivity.getRange(`${col}4:${col}13`).formulas = absHelperFormulas[h];
}

for (const item of metricRows) {
  const { metricRow: row, w, s, scoreCol, scoreRankCol, refCol, refRankCol, helperCol } = item;
  sensitivity.getRange(`A${row}:B${row}`).values = [[windows[w], comparisonNames[s]]];
  sensitivity.getRange(`C${row}:J${row}`).formulas = [[
    `=COUNT(${scoreCol}$4:${scoreCol}$13)`,
    `=AVERAGE(${helperCol}$4:${helperCol}$13)`,
    `=SQRT(SUMXMY2(${scoreCol}$4:${scoreCol}$13,${refCol}$4:${refCol}$13)/COUNT(${scoreCol}$4:${scoreCol}$13))`,
    `=AVERAGE(${scoreCol}$4:${scoreCol}$13)-AVERAGE(${refCol}$4:${refCol}$13)`,
    `=CORREL(${scoreCol}$4:${scoreCol}$13,${refCol}$4:${refCol}$13)`,
    `=CORREL(${scoreRankCol}$4:${scoreRankCol}$13,${refRankCol}$4:${refRankCol}$13)`,
    `=2*COVAR(${scoreCol}$4:${scoreCol}$13,${refCol}$4:${refCol}$13)/(VARP(${scoreCol}$4:${scoreCol}$13)+VARP(${refCol}$4:${refCol}$13)+(AVERAGE(${scoreCol}$4:${scoreCol}$13)-AVERAGE(${refCol}$4:${refCol}$13))^2)`,
    `=1-SUMXMY2(${refCol}$4:${refCol}$13,${scoreCol}$4:${scoreCol}$13)/DEVSQ(${refCol}$4:${refCol}$13)`,
  ]];
}

sensitivity.getRange("A33:J34").values = [[
  "Method note",
  "Centered moving averages are sampled at 0, 300, ..., 2700 sec. Endpoints use available observations only; counts are shown in J:N.",
  null, null, null, null, null, null, null, null,
], [
  "Primary window",
  "11 seconds (±5 sec), matching the approximately 10-second human observation interval.",
  null, null, null, null, null, null, null, null,
]];

// Formula-backed chart summaries.
sensitivity.getRange("A37:D37").values = [["window_seconds", "gold", "bronze", "tadano"]];
sensitivity.getRange("F37:I37").values = [["window_seconds", "gold", "bronze", "tadano"]];
for (let w = 0; w < windows.length; w += 1) {
  const row = 38 + w;
  const metricBase = 16 + w * 3;
  sensitivity.getRange(`A${row}`).values = [[windows[w]]];
  sensitivity.getRange(`B${row}:D${row}`).formulas = [[
    `=G${metricBase}`, `=G${metricBase + 1}`, `=G${metricBase + 2}`,
  ]];
  sensitivity.getRange(`F${row}`).values = [[windows[w]]];
  sensitivity.getRange(`G${row}:I${row}`).formulas = [[
    `=D${metricBase}`, `=D${metricBase + 1}`, `=D${metricBase + 2}`,
  ]];
}
const correlationChart = sensitivity.charts.add("line", sensitivity.getRange("A37:D42"));
correlationChart.title = "Pearson r by Moving-Average Window";
correlationChart.hasLegend = true;
correlationChart.yAxis = { min: -1, max: 1, numberFormatCode: "0.00" };
correlationChart.setPosition("K16", "T30");
const maeChart = sensitivity.charts.add("line", sensitivity.getRange("F37:I42"));
maeChart.title = "MAE by Moving-Average Window";
maeChart.hasLegend = true;
maeChart.yAxis = { min: 0, max: 2, numberFormatCode: "0.00" };
maeChart.setPosition("K32", "T46");

// Apply a restrained style consistent with the source workbook.
for (const sheet of [moving, sensitivity]) {
  sheet.getRange("A1:V1").format = { fill: "#1D3F66", font: { bold: true, color: "#FFFFFF", size: 16 } };
  sheet.getRange("A3:V3").format = { fill: "#DCEBF7", font: { bold: true, color: "#17365D" } };
  sheet.freezePanes.freezeRows(3);
}
moving.getRange("A4:J2711").format.numberFormat = "0.000";
moving.getRange("A4:A2711").format.numberFormat = "0";
moving.getRange("A1:P30").format.autofitColumns();
moving.getRange("A:A").format.columnWidth = 14;
sensitivity.getRange("A4:A13").format.numberFormat = "0";
sensitivity.getRange("B4:I13").format.numberFormat = "0.0000";
sensitivity.getRange("J4:N13").format.numberFormat = "0";
sensitivity.getRange("C16:C30").format.numberFormat = "0";
sensitivity.getRange("D16:J30").format.numberFormat = "0.0000";
sensitivity.getRange("A15:J15").format = { fill: "#4472C4", font: { bold: true, color: "#FFFFFF" } };
sensitivity.getRange("A33:A34").format = { fill: "#FFF2CC", font: { bold: true } };
sensitivity.getRange("B33:J34").merge(true);
sensitivity.getRange("B33:J34").format = { fill: "#FFF2CC", wrapText: true };
sensitivity.getRange("A1:V42").format.autofitColumns();
sensitivity.getRange("A:A").format.columnWidth = 19;
sensitivity.getRange("B:B").format.columnWidth = 21;
sensitivity.getRange("B33:J34").format.rowHeight = 30;

// Update the primary agreement sheet to the 11-sec (±5-sec) analysis.
const agreement = workbook.worksheets.getItem("Agreement Analysis");
agreement.getRange("A1").values = [["Prediction Accuracy & Agreement Analysis (11-sec centered moving average)"]];
agreement.getRange("A3:D3").values = [[
  "evaluation_time (sec)", "window_start (sec)", "observed_seconds", "score_mean_MA_11s",
]];
agreement.getRange("A4:A13").values = evaluationTimes.map((time) => [time]);
agreement.getRange("B4:B13").formulas = evaluationTimes.map((_, i) => [`=MAX(0,A${i + 4}-5)`]);
agreement.getRange("C4:C13").formulas = evaluationTimes.map((_, i) => [`='Window Sensitivity'!K${i + 4}`]);
agreement.getRange("D4:D13").formulas = evaluationTimes.map((_, i) => [`='Window Sensitivity'!C${i + 4}`]);
for (let i = 0; i < evaluationTimes.length; i += 1) {
  const row = i + 4;
  agreement.getRange(`P${row}:AR${row}`).formulas = [[
    `=D${row}-E${row}`, `=ABS(P${row})`, `=P${row}^2`,
    `=D${row}-F${row}`, `=ABS(S${row})`, `=S${row}^2`,
    `=D${row}-G${row}`, `=ABS(V${row})`, `=V${row}^2`,
    `=E${row}-F${row}`, `=ABS(Y${row})`, `=Y${row}^2`,
    `=E${row}-G${row}`, `=ABS(AB${row})`, `=AB${row}^2`,
    `=F${row}-G${row}`, `=ABS(AE${row})`, `=AE${row}^2`,
    `=RANK(D${row},$D$4:$D$13,1)`, `=RANK(E${row},$E$4:$E$13,1)`,
    `=RANK(F${row},$F$4:$F$13,1)`, `=RANK(G${row},$G$4:$G$13,1)`,
    `=AVERAGE(D${row}:G${row})`, null,
    `=A${row}`, `=D${row}`, `=E${row}`, `=F${row}`, `=G${row}`,
  ]];
}
agreement.getRange("A33").values = [["Minimum observed seconds/window"]];
agreement.getRange("B34").formulas = [[
  '=\"Primary comparison uses the 11-sec centered moving average sampled every 300 sec; endpoint windows use available observations only.\"',
]];
const agreementChart = agreement.charts.items[0];
agreementChart.title = "11-sec Moving Average at Human-Rating Times";
agreementChart.series.items[0].name = "score_mean_MA_11s";

await renderTo("after_raw", { sheetName: "Raw Data", range: "A1:J30" });
await renderTo("after_time_series", { sheetName: "Time Series Chart", range: "A1:N30" });
await renderTo("after_agreement", { sheetName: "Agreement Analysis", range: "A1:N43" });
await renderTo("after_moving", { sheetName: "Moving Average Series", range: "A1:AB30" });
await renderTo("after_sensitivity_top", { sheetName: "Window Sensitivity", range: "A1:V34" });
await renderTo("after_sensitivity_charts", { sheetName: "Window Sensitivity", range: "A35:T46" });

for (const query of [
  { kind: "table", sheetId: "Window Sensitivity", range: "A3:N30", include: "values,formulas", tableMaxRows: 30, tableMaxCols: 14, maxChars: 24000 },
  { kind: "table", sheetId: "Agreement Analysis", range: "A3:G24", include: "values,formulas", tableMaxRows: 24, tableMaxCols: 7, maxChars: 16000 },
  { kind: "drawing", sheetId: "Moving Average Series", maxChars: 6000 },
  { kind: "drawing", sheetId: "Window Sensitivity", maxChars: 6000 },
]) {
  const result = await workbook.inspect(query);
  process.stdout.write(result.ndjson + "\n");
}
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
process.stdout.write(errors.ndjson + "\n");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
process.stdout.write(JSON.stringify({ outputPath }) + "\n");
