import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const sourcePath = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/predictions/20260714_unnan_nishi_6-1_2_full/analysis/concentration_prediction_agreement_analysis.xlsx";
const outputDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/20260714_unnan_nishi_6-1_2_full/analysis";
const outputPath = `${outputDir}/concentration_agreement_evaluation_template.xlsx`;
const previewDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/.codex_work/spreadsheet_inspect/previews_agreement_template";
const rawLastRow = 10000;
const chartRawCapacity = 4000;
const processedLastRow = chartRawCapacity + 2;
const evalFirstRow = 4;
const evalLastRow = 13;

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const sourceBook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));
const sourceRaw = sourceBook.worksheets.getItem("Raw Data");
const currentRaw = sourceRaw.getRange("A1:J3315").values.map((row, rowIndex) =>
  row.map((value, colIndex) => {
    if (rowIndex === 0 || colIndex === 0 || value === null || value === "") return value;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : value;
  })
);

const workbook = Workbook.create();
const instructions = workbook.worksheets.add("使い方");
const settings = workbook.worksheets.add("Settings");
const raw = workbook.worksheets.add("Raw Data");
const processed = workbook.worksheets.add("Processed Series");
const evaluation = workbook.worksheets.add("Evaluation");
const metrics = workbook.worksheets.add("Metrics");
const dashboard = workbook.worksheets.add("Dashboard");

for (const sheet of [instructions, settings, raw, processed, evaluation, metrics, dashboard]) {
  sheet.showGridLines = false;
}

// Instructions
instructions.getRange("A1:H1").merge();
instructions.getRange("A1").values = [["集中度予測・人手評価 一致度テンプレート"]];
instructions.getRange("A3:B10").values = [
  ["手順", "内容"],
  ["1", "Raw Dataシートの既存データを、ヘッダーを含めて同じ10列構成の新しいデータに貼り替えます。"],
  ["2", "secはAI予測の秒、score_meanはAI予測値、sec_standardは人手評価時刻です。"],
  ["3", "gold_standard・bronze_standard・tadanoには、sec_standardに対応する人手評価値を入力します。"],
  ["4", "Settingsで局所平均窓、表示用移動平均窓、変化なし判定幅を変更できます。窓幅は奇数を推奨します。"],
  ["5", "Evaluation・Metrics・Dashboardは数式で自動更新されます。原則として直接編集しません。"],
  ["主解析", "AIと人手評価の比較には11秒中央局所平均（±5秒）を使用します。"],
  ["変化傾向", "ΔPearson、ΔSpearman、変化方向一致率は、隣接する人手評価時点間の上下変化を評価します。"],
];
instructions.getRange("A12:B17").values = [
  ["注意事項", "説明"],
  ["端点", "開始・終了付近では、存在する秒だけで中央移動平均を計算します。Evaluationのobserved_rowsで確認できます。"],
  ["重複秒", "Raw Dataに同じsecが複数ある場合、それらのscore_meanを同じ重みで平均します。"],
  ["R²", "R²が負の場合、人手評価の平均値を常に予測する方法より誤差が大きいことを意味します。"],
  ["方向一致率", "偶然でも約50%一致し得るため、Δ相関と併せて判断します。"],
  ["上限", `Raw Dataの統計参照は${rawLastRow - 1}行、表示用グラフは先頭${chartRawCapacity - 1}データ行まで数式を用意しています。超える場合は数式範囲を拡張してください。`],
];

// Editable settings
settings.getRange("A1:D1").merge();
settings.getRange("A1").values = [["Analysis Settings"]];
settings.getRange("A3:C10").values = [
  ["Parameter", "Value", "Description"],
  ["local_window_seconds", 11, "人手評価との主比較に使う中央局所平均窓。11秒なら±5秒。"],
  ["display_ma_points", 31, "時系列グラフ用の中央移動平均点数（テンプレートでは31点固定）。"],
  ["direction_tolerance", 0.1, "変化量の絶対値がこの値未満なら『変化なし』と扱う。"],
  ["max_raw_second", null, "Raw Dataのsec最大値（自動計算）。"],
  ["raw_formula_last_row", rawLastRow, "数式が参照するRaw Data最終行。"],
  ["chart_raw_capacity", chartRawCapacity - 1, "Processed Seriesで用意したRaw Data行数。"],
  ["evaluation_point_count", null, "sec_standardが入力された評価点数（自動計算）。"],
];
settings.getRange("B7").formulas = [[`=MAX('Raw Data'!$B$2:$B$${rawLastRow})`]];
settings.getRange("B10").formulas = [[`=COUNT('Raw Data'!$G$2:$G$101)`]];
settings.getRange("B4:B5").dataValidation = { rule: { type: "whole", operator: "between", formula1: 1, formula2: 301 } };

// Raw input: initial current data, replaceable by the user.
raw.getRange(`A1:J${currentRaw.length}`).values = currentRaw;
raw.freezePanes.freezeRows(1);

// Lightweight display series. Main statistics below continue to use exact seconds.
processed.getRange("A1:F1").merge();
processed.getRange("A1").values = [["Second-by-Second AI Series and Display Moving Average"]];
processed.getRange("A3:F3").values = [["sec", "score_mean_per_sec", "score_mean_display_MA", "gold", "bronze", "tadano"]];
const processedRows = [];
for (let index = 0; index < chartRawCapacity - 1; index += 1) {
  const row = index + 4;
  const rawRow = index + 2;
  const firstDataRow = 4;
  const lastDataRow = processedLastRow;
  const displayHalfWidth = 15;
  const averageStartRow = Math.max(firstDataRow, row - displayHalfWidth);
  const averageEndRow = Math.min(lastDataRow, row + displayHalfWidth);
  processedRows.push([
    `=IF(COUNT('Raw Data'!B${rawRow})=0,\"\",'Raw Data'!B${rawRow})`,
    `=IF(COUNT('Raw Data'!D${rawRow})=0,\"\",'Raw Data'!D${rawRow})`,
    `=IF(B${row}=\"\",\"\",AVERAGE(B${averageStartRow}:B${averageEndRow}))`,
    `=IFERROR(INDEX('Raw Data'!$H$2:$H$101,MATCH(A${row},'Raw Data'!$G$2:$G$101,0)),\"\")`,
    `=IFERROR(INDEX('Raw Data'!$I$2:$I$101,MATCH(A${row},'Raw Data'!$G$2:$G$101,0)),\"\")`,
    `=IFERROR(INDEX('Raw Data'!$J$2:$J$101,MATCH(A${row},'Raw Data'!$G$2:$G$101,0)),\"\")`,
  ]);
}
processed.getRange(`A4:F${processedLastRow}`).formulas = processedRows;
processed.freezePanes.freezeRows(3);

// Human-rating-time local averages and change helpers.
evaluation.getRange("A1:X1").merge();
evaluation.getRange("A1").values = [["Local-Mean Evaluation at Human-Rating Times"]];
evaluation.getRange("A3:X3").values = [[
  "evaluation_time", "observed_rows", "AI_local_mean", "gold", "bronze", "tadano",
  "delta_AI", "delta_gold", "delta_bronze", "delta_tadano",
  "rank_AI", "rank_gold", "rank_bronze", "rank_tadano",
  "rank_delta_AI", "rank_delta_gold", "rank_delta_bronze", "rank_delta_tadano",
  "direction_match_gold", "direction_match_bronze", "direction_match_tadano",
  "abs_error_gold", "abs_error_bronze", "abs_error_tadano",
]];
for (let row = evalFirstRow; row <= evalLastRow; row += 1) {
  const rawRow = row - 2;
  const previous = row - 1;
  evaluation.getRange(`A${row}:X${row}`).formulas = [[
    `=IF(COUNT('Raw Data'!G${rawRow})=0,\"\",'Raw Data'!G${rawRow})`,
    `=IF(COUNT(A${row})=0,\"\",COUNTIFS('Raw Data'!$B$2:$B$${rawLastRow},\">=\"&MAX(0,A${row}-INT('Settings'!$B$4/2)),'Raw Data'!$B$2:$B$${rawLastRow},\"<=\"&A${row}+INT('Settings'!$B$4/2),'Raw Data'!$D$2:$D$${rawLastRow},\">0\"))`,
    `=IF(COUNT(A${row})=0,\"\",IFERROR(SUMIFS('Raw Data'!$D$2:$D$${rawLastRow},'Raw Data'!$B$2:$B$${rawLastRow},\">=\"&MAX(0,A${row}-INT('Settings'!$B$4/2)),'Raw Data'!$B$2:$B$${rawLastRow},\"<=\"&A${row}+INT('Settings'!$B$4/2))/B${row},\"\"))`,
    `=IF(COUNT('Raw Data'!H${rawRow})=0,\"\",'Raw Data'!H${rawRow})`,
    `=IF(COUNT('Raw Data'!I${rawRow})=0,\"\",'Raw Data'!I${rawRow})`,
    `=IF(COUNT('Raw Data'!J${rawRow})=0,\"\",'Raw Data'!J${rawRow})`,
    row === evalFirstRow ? `=\"\"` : `=IF(OR(C${row}=\"\",C${previous}=\"\"),\"\",C${row}-C${previous})`,
    row === evalFirstRow ? `=\"\"` : `=IF(OR(D${row}=\"\",D${previous}=\"\"),\"\",D${row}-D${previous})`,
    row === evalFirstRow ? `=\"\"` : `=IF(OR(E${row}=\"\",E${previous}=\"\"),\"\",E${row}-E${previous})`,
    row === evalFirstRow ? `=\"\"` : `=IF(OR(F${row}=\"\",F${previous}=\"\"),\"\",F${row}-F${previous})`,
    `=IF(COUNT(C${row})=0,\"\",RANK(C${row},$C$${evalFirstRow}:$C$${evalLastRow},1))`,
    `=IF(COUNT(D${row})=0,\"\",RANK(D${row},$D$${evalFirstRow}:$D$${evalLastRow},1))`,
    `=IF(COUNT(E${row})=0,\"\",RANK(E${row},$E$${evalFirstRow}:$E$${evalLastRow},1))`,
    `=IF(COUNT(F${row})=0,\"\",RANK(F${row},$F$${evalFirstRow}:$F$${evalLastRow},1))`,
    `=IF(COUNT(G${row})=0,\"\",RANK(G${row},$G$${evalFirstRow + 1}:$G$${evalLastRow},1))`,
    `=IF(COUNT(H${row})=0,\"\",RANK(H${row},$H$${evalFirstRow + 1}:$H$${evalLastRow},1))`,
    `=IF(COUNT(I${row})=0,\"\",RANK(I${row},$I$${evalFirstRow + 1}:$I$${evalLastRow},1))`,
    `=IF(COUNT(J${row})=0,\"\",RANK(J${row},$J$${evalFirstRow + 1}:$J$${evalLastRow},1))`,
    `=IF(OR(COUNT(G${row})=0,COUNT(H${row})=0),\"\",IF(AND(ABS(G${row})<'Settings'!$B$6,ABS(H${row})<'Settings'!$B$6),1,IF(SIGN(G${row})=SIGN(H${row}),1,0)))`,
    `=IF(OR(COUNT(G${row})=0,COUNT(I${row})=0),\"\",IF(AND(ABS(G${row})<'Settings'!$B$6,ABS(I${row})<'Settings'!$B$6),1,IF(SIGN(G${row})=SIGN(I${row}),1,0)))`,
    `=IF(OR(COUNT(G${row})=0,COUNT(J${row})=0),\"\",IF(AND(ABS(G${row})<'Settings'!$B$6,ABS(J${row})<'Settings'!$B$6),1,IF(SIGN(G${row})=SIGN(J${row}),1,0)))`,
    `=IF(OR(C${row}=\"\",D${row}=\"\"),\"\",ABS(C${row}-D${row}))`,
    `=IF(OR(C${row}=\"\",E${row}=\"\"),\"\",ABS(C${row}-E${row}))`,
    `=IF(OR(C${row}=\"\",F${row}=\"\"),\"\",ABS(C${row}-F${row}))`,
  ]];
}
evaluation.freezePanes.freezeRows(3);

// Metrics: values and change trends.
metrics.getRange("A1:Q1").merge();
metrics.getRange("A1").values = [["Agreement and Change-Trend Metrics"]];
metrics.getRange("A3:Q3").values = [[
  "comparison", "N", "MAE", "RMSE", "Bias", "Pearson r", "Spearman rho", "CCC",
  "R²", "R² (%)", "ΔN", "ΔPearson", "ΔSpearman", "direction matches",
  "direction total", "direction agreement", "direction agreement (%)",
]];
const comparisons = [
  { name: "AI vs gold", ref: "D", rank: "L", delta: "H", deltaRank: "P", match: "S", abs: "V" },
  { name: "AI vs bronze", ref: "E", rank: "M", delta: "I", deltaRank: "Q", match: "T", abs: "W" },
  { name: "AI vs tadano", ref: "F", rank: "N", delta: "J", deltaRank: "R", match: "U", abs: "X" },
];
const levelRange = (col) => `'Evaluation'!$${col}$${evalFirstRow}:$${col}$${evalLastRow}`;
const deltaRange = (col) => `'Evaluation'!$${col}$${evalFirstRow + 1}:$${col}$${evalLastRow}`;
for (let i = 0; i < comparisons.length; i += 1) {
  const row = 4 + i;
  const c = comparisons[i];
  metrics.getRange(`A${row}`).values = [[c.name]];
  metrics.getRange(`B${row}:Q${row}`).formulas = [[
    `=COUNT(${levelRange("C")})`,
    `=AVERAGE(${levelRange(c.abs)})`,
    `=SQRT(SUMXMY2(${levelRange("C")},${levelRange(c.ref)})/B${row})`,
    `=AVERAGE(${levelRange("C")})-AVERAGE(${levelRange(c.ref)})`,
    `=CORREL(${levelRange("C")},${levelRange(c.ref)})`,
    `=CORREL(${levelRange("K")},${levelRange(c.rank)})`,
    `=2*COVAR(${levelRange("C")},${levelRange(c.ref)})/(VARP(${levelRange("C")})+VARP(${levelRange(c.ref)})+(AVERAGE(${levelRange("C")})-AVERAGE(${levelRange(c.ref)}))^2)`,
    `=1-SUMXMY2(${levelRange(c.ref)},${levelRange("C")})/DEVSQ(${levelRange(c.ref)})`,
    `=I${row}*100`,
    `=COUNT(${deltaRange("G")})`,
    `=CORREL(${deltaRange("G")},${deltaRange(c.delta)})`,
    `=CORREL(${deltaRange("O")},${deltaRange(c.deltaRank)})`,
    `=SUM(${deltaRange(c.match)})`,
    `=COUNT(${deltaRange(c.match)})`,
    `=N${row}/O${row}`,
    `=P${row}*100`,
  ]];
}
metrics.getRange("A9:B15").values = [
  ["指標", "解釈"],
  ["Pearson r", "値の上下が線形に対応するほど+1に近い。"],
  ["Spearman rho", "高い時点・低い時点の順位が一致するほど+1に近い。"],
  ["ΔPearson", "隣接評価時点間の変化量が対応するほど+1に近い。"],
  ["ΔSpearman", "変化量の大小順位が一致するほど+1に近い。"],
  ["方向一致率", "AIと人手評価が同じ方向へ変化した区間の割合。"],
  ["CCC / MAE / R²", "値そのものの一致度・誤差。変化傾向とは分けて解釈する。"],
];

// Dashboard and charts.
dashboard.getRange("A1:N1").merge();
dashboard.getRange("A1").values = [["Concentration Agreement Dashboard"]];
dashboard.getRange("A3:Q6").formulas = [
  ["='Metrics'!A3", "='Metrics'!B3", "='Metrics'!C3", "='Metrics'!D3", "='Metrics'!E3", "='Metrics'!F3", "='Metrics'!G3", "='Metrics'!H3", "='Metrics'!I3", "='Metrics'!J3", "='Metrics'!K3", "='Metrics'!L3", "='Metrics'!M3", "='Metrics'!N3", "='Metrics'!O3", "='Metrics'!P3", "='Metrics'!Q3"],
  Array.from({ length: 17 }, (_, i) => `='Metrics'!${String.fromCharCode(65 + i)}4`),
  Array.from({ length: 17 }, (_, i) => `='Metrics'!${String.fromCharCode(65 + i)}5`),
  Array.from({ length: 17 }, (_, i) => `='Metrics'!${String.fromCharCode(65 + i)}6`),
];
dashboard.getRange("A9:F9").values = [["evaluation_time", "AI_local_mean", "gold", "bronze", "tadano", "AI_display_MA_at_time"]];
for (let row = 10; row <= 19; row += 1) {
  const erow = row - 6;
  dashboard.getRange(`A${row}:F${row}`).formulas = [[
    `='Evaluation'!A${erow}`, `='Evaluation'!C${erow}`, `='Evaluation'!D${erow}`,
    `='Evaluation'!E${erow}`, `='Evaluation'!F${erow}`,
    `=IF(COUNT(A${row})=0,\"\",IFERROR(INDEX('Processed Series'!$C$4:$C$${processedLastRow},MATCH(A${row},'Processed Series'!$A$4:$A$${processedLastRow},0)),\"\"))`,
  ]];
}
dashboard.getRange("H9:L9").values = [["evaluation_time", "delta_AI", "delta_gold", "delta_bronze", "delta_tadano"]];
for (let row = 10; row <= 19; row += 1) {
  const erow = row - 6;
  dashboard.getRange(`H${row}:L${row}`).formulas = [[
    `='Evaluation'!A${erow}`, `='Evaluation'!G${erow}`, `='Evaluation'!H${erow}`,
    `='Evaluation'!I${erow}`, `='Evaluation'!J${erow}`,
  ]];
}

const trendChart = dashboard.charts.add("line", { title: "Smoothed AI Trend and Human Ratings", hasLegend: true });
const aiSeries = trendChart.series.add("AI display moving average");
aiSeries.categoryFormula = `'Dashboard'!$A$10:$A$19`;
aiSeries.formula = `'Dashboard'!$F$10:$F$19`;
for (const [name, col, color] of [["gold", "C", "#ED7D31"], ["bronze", "D", "#70AD47"], ["tadano", "E", "#5B9BD5"]]) {
  const series = trendChart.series.add(name);
  series.categoryFormula = `'Dashboard'!$A$10:$A$19`;
  series.formula = `'Dashboard'!$${col}$10:$${col}$19`;
  series.fill = color;
}
trendChart.yAxis = { min: 1, max: 7, numberFormatCode: "0.0" };
trendChart.setPosition("A22", "J42");

const changeChart = dashboard.charts.add("line", dashboard.getRange("H9:L19"));
changeChart.title = "Change Between Human-Rating Times";
changeChart.hasLegend = true;
changeChart.yAxis = { min: -3, max: 3, numberFormatCode: "0.0" };
changeChart.setPosition("K22", "T42");

// Styling
const titleFormat = { fill: "#1D3F66", font: { bold: true, color: "#FFFFFF", size: 16 } };
const headerFormat = { fill: "#DCEBF7", font: { bold: true, color: "#17365D" } };
const strongHeader = { fill: "#4472C4", font: { bold: true, color: "#FFFFFF" } };
instructions.getRange("A1:H1").format = titleFormat;
settings.getRange("A1:D1").format = titleFormat;
processed.getRange("A1:F1").format = titleFormat;
evaluation.getRange("A1:X1").format = titleFormat;
metrics.getRange("A1:Q1").format = titleFormat;
dashboard.getRange("A1:N1").format = titleFormat;
instructions.getRange("A3:B3").format = strongHeader;
instructions.getRange("A12:B12").format = strongHeader;
settings.getRange("A3:C3").format = strongHeader;
raw.getRange("A1:J1").format = { fill: "#1D3F66", font: { bold: true, color: "#FFFFFF" } };
processed.getRange("A3:F3").format = headerFormat;
evaluation.getRange("A3:X3").format = headerFormat;
metrics.getRange("A3:Q3").format = strongHeader;
metrics.getRange("A9:B9").format = strongHeader;
dashboard.getRange("A3:Q3").format = strongHeader;
dashboard.getRange("A9:F9").format = headerFormat;
dashboard.getRange("H9:L9").format = headerFormat;
settings.getRange("B4:B6").format = { fill: "#FFF2CC", font: { bold: true, color: "#9C6500" } };

instructions.getRange("A1:H17").format.wrapText = true;
instructions.getRange("A:A").format.columnWidth = 16;
instructions.getRange("B:B").format.columnWidth = 90;
instructions.getRange("A3:B17").format.autofitRows();
settings.getRange("A:A").format.columnWidth = 28;
settings.getRange("B:B").format.columnWidth = 14;
settings.getRange("C:C").format.columnWidth = 68;
settings.getRange("C3:C10").format.wrapText = true;
raw.getRange("A1:J40").format.autofitColumns();
processed.getRange("A:F").format.columnWidth = 24;
evaluation.getRange("A:X").format.columnWidth = 17;
metrics.getRange("A:Q").format.columnWidth = 17;
metrics.getRange("A:A").format.columnWidth = 22;
metrics.getRange("B9:B15").format.columnWidth = 68;
metrics.getRange("B9:B15").format.wrapText = true;
dashboard.getRange("A:Q").format.columnWidth = 15;
dashboard.getRange("A:A").format.columnWidth = 20;
dashboard.getRange("A10:F19").format.numberFormat = "0.0000";
dashboard.getRange("H10:L19").format.numberFormat = "0.0000";
evaluation.getRange(`A4:B${evalLastRow}`).format.numberFormat = "0";
evaluation.getRange(`C4:X${evalLastRow}`).format.numberFormat = "0.0000";
metrics.getRange("B4:O6").format.numberFormat = "0.0000";
metrics.getRange("P4:P6").format.numberFormat = "0.0%";
metrics.getRange("Q4:Q6").format.numberFormat = "0.0";
dashboard.getRange("B4:O6").format.numberFormat = "0.0000";
dashboard.getRange("P4:P6").format.numberFormat = "0.0%";
dashboard.getRange("Q4:Q6").format.numberFormat = "0.0";
processed.getRange(`A4:A${processedLastRow}`).format.numberFormat = "0";
processed.getRange(`B4:F${processedLastRow}`).format.numberFormat = "0.0000";

await workbook.inspect({ kind: "sheet", include: "id,name" });

async function saveRender(name, sheetName, range) {
  const image = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await image.arrayBuffer()));
}
await saveRender("instructions", "使い方", "A1:H17");
await saveRender("settings", "Settings", "A1:D10");
await saveRender("raw", "Raw Data", "A1:J30");
await saveRender("processed", "Processed Series", "A1:F30");
await saveRender("evaluation", "Evaluation", "A1:X16");
await saveRender("metrics", "Metrics", "A1:Q15");
await saveRender("dashboard_top", "Dashboard", "A1:Q18");
await saveRender("dashboard_charts", "Dashboard", "A20:T43");

for (const query of [
  { kind: "table", sheetId: "Settings", range: "A3:C10", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 3, maxChars: 6000 },
  { kind: "table", sheetId: "Evaluation", range: "A3:X14", include: "values,formulas", tableMaxRows: 14, tableMaxCols: 24, maxChars: 20000 },
  { kind: "table", sheetId: "Metrics", range: "A3:Q6", include: "values,formulas", tableMaxRows: 6, tableMaxCols: 17, maxChars: 12000 },
  { kind: "drawing", sheetId: "Dashboard", maxChars: 6000 },
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
