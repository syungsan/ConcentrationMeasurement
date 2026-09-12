import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/019f8ee8-9b86-7b50-881b-60f28d53a92b/agg_timeseries_sec.csv";
const outputDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/019f8ee8-9b86-7b50-881b-60f28d53a92b";
const outputPath = `${outputDir}/concentration_prediction_agreement_analysis_compatible.xlsx`;
const previewPath = `${outputDir}/final_timeseries_preview.png`;
const analysisPreviewPath = `${outputDir}/final_agreement_preview.png`;

const csvText = await fs.readFile(sourcePath, "utf8");
const parsed = csvText.replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean).map((line) => line.split(","));
const headers = parsed[0];
const secCol = headers.indexOf("sec");
const scoreCol = headers.indexOf("score_mean");
const stdSecCol = headers.indexOf("sec_standard");
const goldCol = headers.includes("golden_standard") ? headers.indexOf("golden_standard") : headers.indexOf("gold_standard");
const bronzeCol = headers.indexOf("bronze_standard");
const scoreBuckets = new Map();
const standards = new Map();
for (const row of parsed.slice(1)) {
  const sec = Number(row[secCol]);
  const score = Number(row[scoreCol]);
  if (Number.isFinite(sec) && Number.isFinite(score)) {
    const bucket = scoreBuckets.get(sec) ?? [];
    bucket.push(score);
    scoreBuckets.set(sec, bucket);
  }
  const stdSec = Number(row[stdSecCol]);
  const gold = Number(row[goldCol]);
  const bronze = Number(row[bronzeCol]);
  if (row[stdSecCol] !== "" && Number.isFinite(stdSec) && Number.isFinite(gold) && Number.isFinite(bronze)) {
    standards.set(stdSec, [gold, bronze]);
  }
}
if (standards.size === 0) {
  const standardValues = [
    [0, 4, 4.844444444], [300, 4.444444444, 4.955555556], [600, 5.888888889, 5.777777778],
    [900, 4.777777778, 4.466666667], [1200, 4.555555556, 4.466666667], [1500, 5.111111111, 5.333333333],
    [1800, 5.111111111, 5.844444444], [2100, 5.222222222, 5], [2400, 5.333333333, 5.777777778],
    [2700, 5.888888889, 6.222222222],
  ];
  for (const [sec, gold, bronze] of standardValues) standards.set(sec, [gold, bronze]);
}
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Raw Data" });
const raw = workbook.worksheets.getItem("Raw Data");
raw.name = "Raw Data";

const analysis = workbook.worksheets.add("Time Series Chart");
analysis.showGridLines = false;

analysis.getRange("A1:H1").merge();
analysis.getRange("A1").values = [["Concentration Score Time Series"]];
analysis.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
analysis.getRange("A1:H1").format.rowHeight = 30;

analysis.getRange("A3:D3").values = [["time (sec)", "score_mean", "gold_standard", "bronze_standard"]];
analysis.getRange("A3:D3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  borders: { preset: "outside", style: "thin", color: "#8EA9C1" },
};

const maxSec = 2707;
const standardTimes = [...standards.keys()].sort((a, b) => a - b);
function interpolatedStandard(sec, index) {
  if (sec <= standardTimes[0]) return standards.get(standardTimes[0])[index];
  if (sec >= standardTimes.at(-1)) return standards.get(standardTimes.at(-1))[index];
  const right = standardTimes.find((t) => t >= sec);
  const left = standardTimes[standardTimes.indexOf(right) - 1];
  const leftValue = standards.get(left)[index];
  const rightValue = standards.get(right)[index];
  return leftValue + (rightValue - leftValue) * ((sec - left) / (right - left));
}
const rows = Array.from({ length: maxSec + 1 }, (_, sec) => {
  const bucket = scoreBuckets.get(sec) ?? [];
  const score = bucket.length ? bucket.reduce((a, b) => a + b, 0) / bucket.length : null;
  return [sec, score, interpolatedStandard(sec, 0), interpolatedStandard(sec, 1)];
});
analysis.getRange(`A4:D${maxSec + 4}`).values = rows;

analysis.getRange(`A4:A${maxSec + 4}`).format.numberFormat = "0";
analysis.getRange(`B4:D${maxSec + 4}`).format.numberFormat = "0.000";
analysis.getRange("A:D").format.columnWidth = 16;
analysis.freezePanes.freezeRows(3);

const chart = analysis.charts.add("scatter", analysis.getRange(`A3:D${maxSec + 4}`));
chart.name = "Score Standards Scatter Time Series";
chart.title = "score_mean と基準値の散布図（共通 time (sec) 軸）";
chart.titleTextStyle.fontSize = 13;
chart.hasLegend = true;
chart.legend.position = "bottom";
chart.xAxis = { numberFormatCode: "0", min: 0, max: 2707, majorUnit: 300, textStyle: { fontSize: 9 } };
chart.xAxis.title.text = "time (sec)";
chart.yAxis = { numberFormatCode: "0.0", min: 0, max: 7, majorUnit: 1 };
chart.yAxis.title.text = "score";
chart.setPosition("F3", "Q25");

const series = chart.series.items;
if (series[0]) {
  series[0].chartType = "XYScatterLines";
  series[0].line = { color: "#2F75B5", width: 0.9 };
  series[0].markerStyle = "circle";
  series[0].markerSize = 1;
  series[0].markerBackgroundColor = "#2F75B5";
  series[0].markerForegroundColor = "#2F75B5";
}
if (series[1]) {
  series[1].chartType = "XYScatterLinesNoMarkers";
  series[1].line = { color: "#ED7D31", width: 0.55 };
  series[1].markerStyle = "none";
  series[1].markerSize = 1;
  series[1].markerBackgroundColor = "#ED7D31";
  series[1].markerForegroundColor = "#ED7D31";
}
if (series[2]) {
  series[2].chartType = "XYScatterLinesNoMarkers";
  series[2].line = { color: "#A5A5A5", width: 0.55 };
  series[2].markerStyle = "none";
  series[2].markerSize = 1;
  series[2].markerBackgroundColor = "#A5A5A5";
  series[2].markerForegroundColor = "#A5A5A5";
}

analysis.getRange("F27:Q28").merge();
analysis.getRange("F27").values = [["注: 同じ秒に複数の score_mean がある場合は平均値を表示。基準値は sec_standard の300秒間隔の原値を直線で結んで表示。"]];
analysis.getRange("F27:Q28").format = {
  fill: "#F3F6F9",
  font: { color: "#44546A", italic: true, size: 9 },
  wrapText: true,
  verticalAlignment: "center",
};

const agreement = workbook.worksheets.add("Agreement Analysis");
agreement.showGridLines = false;
agreement.getRange("A1:H1").merge();
agreement.getRange("A1").values = [["Prediction Accuracy & Agreement Analysis"]];
agreement.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
agreement.getRange("A1:H1").format.rowHeight = 30;
agreement.getRange("A3:F3").values = [["bin_start (sec)", "bin_end (sec)", "observed_seconds", "score_mean_300s", "gold_standard", "bronze_standard"]];
agreement.getRange("A3:F3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  borders: { preset: "outside", style: "thin", color: "#8EA9C1" },
};

const binRows = standardTimes.map((start, index) => {
  const end = index < standardTimes.length - 1 ? standardTimes[index + 1] - 1 : maxSec;
  const values = [];
  for (let sec = start; sec <= end; sec++) {
    const bucket = scoreBuckets.get(sec);
    if (bucket?.length) values.push(bucket.reduce((a, b) => a + b, 0) / bucket.length);
  }
  const score = values.reduce((a, b) => a + b, 0) / values.length;
  const std = standards.get(start);
  return [start, end, values.length, score, std[0], std[1]];
});
agreement.getRange("A4:F13").values = binRows;
agreement.getRange("A4:C13").format.numberFormat = "0";
agreement.getRange("D4:F13").format.numberFormat = "0.000";

agreement.getRange("P3:AB3").values = [["score-gold", "abs(score-gold)", "sq(score-gold)", "rank_score", "rank_gold", "score-bronze", "abs(score-bronze)", "sq(score-bronze)", "rank_bronze", "gold-bronze", "abs(gold-bronze)", "sq(gold-bronze)", "row_mean"]];
agreement.getRange("P3:AB3").format = { fill: "#E2F0D9", font: { bold: true, color: "#385723" } };
agreement.getRange("P4:AB4").formulas = [[
  "=D4-E4", "=ABS(P4)", "=P4^2", "=RANK(D4,$D$4:$D$13,1)", "=RANK(E4,$E$4:$E$13,1)",
  "=D4-F4", "=ABS(U4)", "=U4^2", "=RANK(F4,$F$4:$F$13,1)", "=E4-F4", "=ABS(Y4)", "=Y4^2", "=AVERAGE(D4:F4)"
]];
agreement.getRange("P4:AB13").fillDown();
agreement.getRange("P4:AB13").format.numberFormat = "0.0000";

agreement.getRange("A16:D16").values = [["Metric", "score_mean vs gold", "score_mean vs bronze", "gold vs bronze"]];
agreement.getRange("A16:D16").format = { fill: "#4472C4", font: { bold: true, color: "#FFFFFF" } };
agreement.getRange("A17:A24").values = [["N"], ["MAE"], ["RMSE"], ["Bias (first - second)"], ["Pearson r"], ["Spearman rho"], ["CCC"], ["R² (prediction agreement)"]];
agreement.getRange("B17:D24").formulas = [
  ["=COUNT(D4:D13)", "=COUNT(D4:D13)", "=COUNT(E4:E13)"],
  ["=AVERAGE(Q4:Q13)", "=AVERAGE(V4:V13)", "=AVERAGE(Z4:Z13)"],
  ["=SQRT(AVERAGE(R4:R13))", "=SQRT(AVERAGE(W4:W13))", "=SQRT(AVERAGE(AA4:AA13))"],
  ["=AVERAGE(P4:P13)", "=AVERAGE(U4:U13)", "=AVERAGE(Y4:Y13)"],
  ["=CORREL(D4:D13,E4:E13)", "=CORREL(D4:D13,F4:F13)", "=CORREL(E4:E13,F4:F13)"],
  ["=CORREL(S4:S13,T4:T13)", "=CORREL(S4:S13,X4:X13)", "=CORREL(T4:T13,X4:X13)"],
  ["=2*COVAR(D4:D13,E4:E13)/(VARP(D4:D13)+VARP(E4:E13)+(AVERAGE(D4:D13)-AVERAGE(E4:E13))^2)", "=2*COVAR(D4:D13,F4:F13)/(VARP(D4:D13)+VARP(F4:F13)+(AVERAGE(D4:D13)-AVERAGE(F4:F13))^2)", "=2*COVAR(E4:E13,F4:F13)/(VARP(E4:E13)+VARP(F4:F13)+(AVERAGE(E4:E13)-AVERAGE(F4:F13))^2)"],
  ["=1-SUMXMY2(E4:E13,D4:D13)/DEVSQ(E4:E13)", "=1-SUMXMY2(F4:F13,D4:D13)/DEVSQ(F4:F13)", "=1-SUMXMY2(F4:F13,E4:E13)/DEVSQ(F4:F13)"]
];
agreement.getRange("B17:D17").format.numberFormat = "0";
agreement.getRange("B18:D24").format.numberFormat = "0.0000";

agreement.getRange("A27:B27").values = [["Three-series agreement", "Value"]];
agreement.getRange("A27:B27").format = { fill: "#70AD47", font: { bold: true, color: "#FFFFFF" } };
agreement.getRange("A28:A34").values = [["Grand mean"], ["MS rows"], ["MS columns"], ["MS error"], ["ICC(2,1) absolute agreement"], ["Minimum observed seconds/bin"], ["Evaluation note"]];
agreement.getRange("B28:B34").formulas = [
  ["=AVERAGE(D4:F13)"],
  ["=3*DEVSQ(AB4:AB13)/9"],
  ["=10*DEVSQ(D15:F15)/2"],
  ["=(DEVSQ(D4:F13)-3*DEVSQ(AB4:AB13)-10*DEVSQ(D15:F15))/18"],
  ["=(B29-B31)/(B29+2*B31+3*(B30-B31)/10)"],
  ["=MIN(C4:C13)"],
  ["=\"300秒区間で秒ごとのscore平均を再平均。最終区間は2700～2707秒のため観測秒数が少ない。\""]
];
agreement.getRange("D15:F15").formulas = [["=AVERAGE(D4:D13)", "=AVERAGE(E4:E13)", "=AVERAGE(F4:F13)"]];
agreement.getRange("B28:B33").format.numberFormat = "0.0000";
agreement.getRange("B34:F35").merge();
agreement.getRange("B34:F35").format = { fill: "#FFF2CC", wrapText: true, verticalAlignment: "center" };

agreement.getRange("A37:G37").values = [["Comparison", "Mean", "Difference", "Bias", "Lower 95% limit", "Upper 95% limit", "Interpretation"]];
agreement.getRange("A37:G37").format = { fill: "#A5A5A5", font: { bold: true, color: "#FFFFFF" } };
agreement.getRange("A38:A40").values = [["score vs gold"], ["score vs bronze"], ["gold vs bronze"]];
agreement.getRange("B38:F40").formulas = [
  ["=AVERAGE(D4:D13,E4:E13)", "=AVERAGE(P4:P13)", "=AVERAGE(P4:P13)", "=D38-1.96*STDEV(P4:P13)", "=D38+1.96*STDEV(P4:P13)"],
  ["=AVERAGE(D4:D13,F4:F13)", "=AVERAGE(U4:U13)", "=AVERAGE(U4:U13)", "=D39-1.96*STDEV(U4:U13)", "=D39+1.96*STDEV(U4:U13)"],
  ["=AVERAGE(E4:E13,F4:F13)", "=AVERAGE(Y4:Y13)", "=AVERAGE(Y4:Y13)", "=D40-1.96*STDEV(Y4:Y13)", "=D40+1.96*STDEV(Y4:Y13)"]
];
agreement.getRange("G38:G40").values = [["Bland–Altman limits"], ["Bland–Altman limits"], ["Bland–Altman limits"]];
agreement.getRange("B38:F40").format.numberFormat = "0.0000";

agreement.getRange("AD3:AG3").values = [["bin_start (sec)", "score_mean_300s", "gold_standard", "bronze_standard"]];
agreement.getRange("AD4:AG4").formulas = [["=A4", "=D4", "=E4", "=F4"]];
agreement.getRange("AD4:AG13").fillDown();
const compareChart = agreement.charts.add("line", agreement.getRange("AD3:AG13"));
compareChart.title = "300秒区間での3系列比較";
compareChart.hasLegend = true;
compareChart.legend.position = "bottom";
compareChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
compareChart.yAxis = { min: 0, max: 7, majorUnit: 1, numberFormatCode: "0.0" };
compareChart.setPosition("G16", "N35");

agreement.getRange("A:T").format.columnWidth = 15;
agreement.getRange("A:A").format.columnWidth = 25;
agreement.getRange("B:D").format.columnWidth = 24;
agreement.getRange("G:G").format.columnWidth = 20;
agreement.getRange("A34:N35").format.rowHeight = 26;
agreement.freezePanes.freezeRows(3);

raw.getRange("A1:M1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" } };
raw.getRange("A:M").format.columnWidth = 15;
raw.getRange("A:A").format.columnWidth = 12;
raw.freezePanes.freezeRows(1);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({ sheetName: "Time Series Chart", range: "A1:Q28", scale: 1.2, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const analysisPreview = await workbook.render({ sheetName: "Agreement Analysis", range: "A1:N40", scale: 1.15, format: "png" });
await fs.writeFile(analysisPreviewPath, new Uint8Array(await analysisPreview.arrayBuffer()));

const inspect = await workbook.inspect({ kind: "table", range: "Time Series Chart!A1:D15", include: "values,formulas", tableMaxRows: 15, tableMaxCols: 4 });
console.log(inspect.ndjson);
const drawings = await workbook.inspect({ kind: "drawing", sheetId: "Time Series Chart", maxChars: 4000 });
console.log(drawings.ndjson);
const metrics = await workbook.inspect({ kind: "table", range: "Agreement Analysis!A16:D34", include: "values,formulas", tableMaxRows: 25, tableMaxCols: 8 });
console.log(metrics.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
console.log(errors.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(`OUTPUT=${outputPath}`);
console.log(`PREVIEW=${previewPath}`);
