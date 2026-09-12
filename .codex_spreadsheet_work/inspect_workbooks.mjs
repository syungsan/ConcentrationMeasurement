import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.resolve("..");
const inputs = {
  template: path.join(root, "outputs", "concentration_agreement_evaluation_template.xlsx"),
  actual: path.join(root, "outputs", "concentration_agreement_evaluation.xlsx"),
};
const renderDir = path.join(process.cwd(), "renders");
await fs.mkdir(renderDir, { recursive: true });

const requested = process.argv[2];
for (const [label, inputPath] of Object.entries(inputs)) {
  if (requested && requested !== label) continue;
  console.log(`=== ${label}: importing ${inputPath} ===`);
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  console.log(`=== ${label}: imported ===`);
  console.log(`=== ${label}: summary ===`);
  console.log((await wb.inspect({
    kind: "workbook,sheet,table,definedName,drawing",
    maxChars: 16000,
    tableMaxRows: 5,
    tableMaxCols: 12,
  })).ndjson);
  console.log(`=== ${label}: formulas ===`);
  console.log((await wb.inspect({
    kind: "formula",
    maxChars: 30000,
    options: { maxResults: 500 },
  })).ndjson);
  console.log(`=== ${label}: errors ===`);
  console.log((await wb.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#CALC!|#SPILL!",
    options: { useRegex: true, maxResults: 500 },
    maxChars: 20000,
  })).ndjson);

  const sheetInfo = await wb.inspect({ kind: "sheet", include: "id,name", maxChars: 10000 });
  for (const line of sheetInfo.ndjson.split(/\r?\n/).filter(Boolean)) {
    const record = JSON.parse(line);
    const sheetName = record.name;
    if (!sheetName) continue;
    try {
      const preview = await wb.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
      const safeName = sheetName.replace(/[\\/:*?"<>|]/g, "_");
      await fs.writeFile(
        path.join(renderDir, `${label}_${safeName}.png`),
        new Uint8Array(await preview.arrayBuffer()),
      );
    } catch (error) {
      console.log(`RENDER ERROR ${label}/${sheetName}: ${error}`);
    }
  }
}
