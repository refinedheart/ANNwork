$ErrorActionPreference = "Stop"

if (-not $env:ANN_DATA_PATH) {
    $env:ANN_DATA_PATH = "C:\Users\PC\Desktop\University\Semester4\Parallel\work510\ANN\ANN\ann\anndata"
}

function Run-And-Save {
    param(
        [string]$QueryLimit,
        [string]$OutputName
    )

    $env:ANN_QUERY_LIMIT = $QueryLimit
    $env:ANN_PQ_RERANK = "550"
    $env:ANN_IVF_NLIST = "128"
    $env:ANN_IVF_NPROBE = "32"
    $env:ANN_HNSW_EF = "64"

    powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\collect_compare_results.ps1"

    $outDir = Join-Path $PSScriptRoot "results"
    Copy-Item (Join-Path $outDir "cross_platform_x86.csv") (Join-Path $outDir $OutputName) -Force
    Write-Host "Saved $OutputName"
}

Run-And-Save -QueryLimit "20" -OutputName "cross_platform_x86_q20_hnsw_simd.csv"
Run-And-Save -QueryLimit "10" -OutputName "cross_platform_x86_q10_hnsw_simd.csv"
Run-And-Save -QueryLimit "200" -OutputName "cross_platform_x86_q200_hnsw_simd.csv"
