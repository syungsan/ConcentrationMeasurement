import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const base = "D:/MyProjects/python/concentration_measurement/concentration_measurement/outputs/predictions/20260714_unnan_nishi_6-1_2_full";
const previewDir = "D:/MyProjects/python/concentration_measurement/concentration_measurement/.codex_work/spreadsheet_inspect/target_chart_previews";
const names = [
  "concentration_agreement_evaluation.xlsx",
  "concentration_agreement_evaluation2.xlsx",
];

await fs.mkdir(previewDir, { recursive: true });

for (const name of names) {
  const path = `${base}/${name}`;
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const sheet = workbook.worksheets.getItem("Processed Series");
  const usedValues = sheet.getUsedRange().values;
  let lastRow = 4;
  for (let index = 3; index < usedValues.length; index += 1) {
    const sec = usedValues[index]?.[0];
    const movingAverage = usedValues[index]?.[2];
    if (typeof sec === "number" && typeof movingAverage === "number") lastRow = index + 1;
  }

  sheet.charts.deleteAll();
  const chart = sheet.charts.add("line", {
    title: "全期間の集中度推移（移動平均）",
    hasLegend: false,
  });
  const series = chart.series.add("score_mean_display_MA");
  series.categoryFormula = `'Processed Series'!$A$4:$A$${lastRow}`;
  series.formula = `'Processed Series'!$C$4:$C$${lastRow}`;
  series.fill = "#1F6D8C";
  chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
  chart.yAxis = { min: 1, max: 7, numberFormatCode: "0.0", textStyle: { fontSize: 9 } };
  chart.setPosition("H2", "T25");

  sheet.getRange("H27:T27").merge();
  sheet.getRange("H27").values = [["横軸：経過秒　縦軸：集中度　系列：score_mean_display_MA"]];
  sheet.getRange("H27:T27").format = {
    fill: "#DCEBF7",
    font: { bold: true, color: "#17365D" },
    horizontalAlignment: "center",
  };

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: `${name} formula error scan`,
  });
  process.stdout.write(`${name}\n${errors.ndjson}\n`);
  process.stdout.write((await workbook.inspect({ kind: "drawing", sheetId: "Processed Series", maxChars: 5000 })).ndjson + "\n");

  const preview = await workbook.render({ sheetName: "Processed Series", range: "A1:T29", scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name.replace(".xlsx", ".png")}`, new Uint8Array(await preview.arrayBuffer()));

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(path);
  process.stdout.write(JSON.stringify({ saved: path, lastRow }) + "\n");
}
