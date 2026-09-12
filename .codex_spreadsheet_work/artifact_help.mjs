import { Workbook } from "@oai/artifact-tool";

const wb = Workbook.create();
wb.worksheets.add("Help");
console.log(wb.help("*", {
  search: "importXlsx|calculation|recalculate",
  include: "index,examples,notes",
  maxChars: 6000,
}).ndjson);
