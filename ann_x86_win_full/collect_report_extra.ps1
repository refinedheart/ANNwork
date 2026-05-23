$ErrorActionPreference = "Stop"

if (-not $env:ANN_DATA_PATH) {
    $env:ANN_DATA_PATH = "C:\Users\PC\Desktop\University\Semester4\Parallel\work510\ANN\ANN\ann\anndata"
}

$exe = Join-Path $PSScriptRoot "build\ann_x86_win.exe"
if (-not (Test-Path $exe)) {
    $exe = Join-Path $PSScriptRoot "build\Release\ann_x86_win.exe"
}
if (-not (Test-Path $exe)) {
    throw "Executable not found. Build first with build_mingw.bat or cmake --build build -j."
}

$outDir = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Get-AnnField {
    param(
        [object[]]$Output,
        [string]$Pattern
    )
    $match = @($Output | Select-String $Pattern | Select-Object -Last 1)
    if ($match.Count -eq 0) {
        throw "Cannot parse '$Pattern' from program output."
    }
    return (($match[0].Line -split ":\s+")[1] -split "\s+")[0]
}

function Invoke-AnnRun {
    param(
        [string]$CsvPath,
        [string]$Suite,
        [string]$Mode,
        [int]$Threads,
        [int]$QueryLimit,
        [string]$ParamName,
        [string]$ParamValue,
        [int]$PqRerank = 550,
        [int]$IvfNprobe = 32,
        [int]$HnswEf = 64
    )

    $env:ANN_QUERY_LIMIT = "$QueryLimit"
    $env:ANN_PQ_RERANK = "$PqRerank"
    $env:ANN_IVF_NLIST = "128"
    $env:ANN_IVF_NPROBE = "$IvfNprobe"
    $env:ANN_HNSW_EF = "$HnswEf"

    Write-Host "=== suite=$Suite mode=$Mode threads=$Threads $ParamName=$ParamValue q=$QueryLimit ==="
    $output = & $exe $Mode $Threads 2>&1
    $output | ForEach-Object { Write-Host $_ }

    $recall = Get-AnnField -Output $output -Pattern "average recall:"
    $latency = Get-AnnField -Output $output -Pattern "average latency"
    $total = Get-AnnField -Output $output -Pattern "total time"

    "$Suite,$Mode,$Threads,$QueryLimit,$ParamName,$ParamValue,$recall,$latency,$total" |
        Add-Content -Encoding UTF8 $CsvPath
}

function New-ResultCsv {
    param([string]$Name)
    $path = Join-Path $outDir $Name
    "suite,mode,threads,query_limit,param_name,param_value,avg_recall,avg_latency_us,total_time_us" |
        Set-Content -Encoding UTF8 $path
    return $path
}

$queryLimit = 200

$simdCsv = New-ResultCsv -Name "x86_extra_simd_q200.csv"
foreach ($mode in @("serial", "sse4", "sse8", "sse16", "avx8", "avx16", "auto")) {
    Invoke-AnnRun -CsvPath $simdCsv -Suite "simd" -Mode $mode -Threads 1 `
        -QueryLimit $queryLimit -ParamName "kernel" -ParamValue $mode
}

$threadCsv = New-ResultCsv -Name "x86_extra_thread_sweep_q200.csv"
foreach ($threads in @(1, 2, 4, 8)) {
    foreach ($mode in @("thread_ivf_pq", "openmp_ivf_pq", "thread_hnsw", "openmp_hnsw")) {
        Invoke-AnnRun -CsvPath $threadCsv -Suite "thread_sweep" -Mode $mode -Threads $threads `
            -QueryLimit $queryLimit -ParamName "threads" -ParamValue "$threads"
    }
}

$tradeoffCsv = New-ResultCsv -Name "x86_extra_tradeoff_q200.csv"
foreach ($nprobe in @(8, 16, 32, 64)) {
    Invoke-AnnRun -CsvPath $tradeoffCsv -Suite "ivf_pq_nprobe" -Mode "ivf_pq" -Threads 1 `
        -QueryLimit $queryLimit -ParamName "ANN_IVF_NPROBE" -ParamValue "$nprobe" -IvfNprobe $nprobe
}
foreach ($ef in @(16, 32, 64, 128)) {
    Invoke-AnnRun -CsvPath $tradeoffCsv -Suite "hnsw_ef" -Mode "hnsw" -Threads 1 `
        -QueryLimit $queryLimit -ParamName "ANN_HNSW_EF" -ParamValue "$ef" -HnswEf $ef
}

Write-Host ""
Write-Host "Saved extra report results:"
Write-Host "  $simdCsv"
Write-Host "  $threadCsv"
Write-Host "  $tradeoffCsv"
