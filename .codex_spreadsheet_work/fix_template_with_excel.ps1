$ErrorActionPreference = 'Stop'

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$template = Join-Path $workspace 'outputs\concentration_agreement_evaluation_template.xlsx'
$backupDir = Join-Path $workspace 'outputs\template_backups'
$backup = Join-Path $backupDir 'concentration_agreement_evaluation_template_before_header_lookup.xlsx'
$pdfDir = Join-Path $workspace 'outputs\template_verification_pdfs'

New-Item -ItemType Directory -Force -Path $backupDir, $pdfDir | Out-Null
if (-not (Test-Path -LiteralPath $backup)) {
    Copy-Item -LiteralPath $template -Destination $backup
}

$excel = $null
$workbook = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.AskToUpdateLinks = $false

    $workbook = $excel.Workbooks.Open($template, 0, $false)
    $evaluation = $workbook.Worksheets.Item('Evaluation')

    $mapping = @{
        'D' = 'gold_standard'
        'E' = 'bronze_standard'
        'F' = 'tadano'
    }
    foreach ($column in $mapping.Keys) {
        $header = $mapping[$column]
        $formula = '=IFERROR(IF(COUNT(INDEX(''Raw Data''!$A$1:$R$10000,ROW()-2,MATCH("' + $header + '",''Raw Data''!$A$1:$R$1,0)))=0,"",INDEX(''Raw Data''!$A$1:$R$10000,ROW()-2,MATCH("' + $header + '",''Raw Data''!$A$1:$R$1,0))),"")'
        $evaluation.Range("${column}4").Formula = $formula
        $evaluation.Range("${column}4:${column}13").FillDown()
    }

    $workbook.Application.CalculateFullRebuild()
    $workbook.Save()

    Get-ChildItem -LiteralPath $pdfDir -Filter '*.pdf' -ErrorAction SilentlyContinue | Remove-Item -Force
    foreach ($sheet in $workbook.Worksheets) {
        $safeName = ($sheet.Name -replace '[\\/:*?"<>|]', '_')
        $pdfPath = Join-Path $pdfDir ($safeName + '.pdf')
        $sheet.ExportAsFixedFormat(0, $pdfPath)
    }

    $errors = @()
    foreach ($sheet in $workbook.Worksheets) {
        $used = $sheet.UsedRange
        foreach ($token in @('#REF!', '#DIV/0!', '#VALUE!', '#NAME?', '#N/A')) {
            $found = $used.Find($token)
            if ($null -ne $found) {
                $errors += "$($sheet.Name)!$($found.Address(0,0))=$token"
            }
        }
    }
    $chartCount = 0
    foreach ($sheet in $workbook.Worksheets) {
        $chartCount += $sheet.ChartObjects().Count
    }
    Write-Output "OUTPUT=$template"
    Write-Output "BACKUP=$backup"
    Write-Output "ERRORS=$($errors -join ';')"
    Write-Output "CHARTS=$chartCount"
    Write-Output "EVALUATION_D4=$($evaluation.Range('D4').Value2)"
    Write-Output "EVALUATION_E4=$($evaluation.Range('E4').Value2)"
    Write-Output "EVALUATION_F4=$($evaluation.Range('F4').Value2)"
}
finally {
    if ($null -ne $workbook) {
        $workbook.Close($true)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($workbook)
    }
    if ($null -ne $excel) {
        $excel.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
