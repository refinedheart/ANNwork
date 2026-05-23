$ErrorActionPreference = "Stop"

$exe = Join-Path $PSScriptRoot "build\ann_x86_win.exe"
if (-not (Test-Path $exe)) {
    $exe = Join-Path $PSScriptRoot "build\Release\ann_x86_win.exe"
}
if (-not (Test-Path $exe)) {
    throw "Executable not found. Build first with build_mingw.bat or build_msvc.bat."
}

$outDir = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outCsv = Join-Path $outDir "cross_platform_x86.csv"

"mode,threads,avg_recall,avg_latency_us,total_time_us" | Set-Content -Encoding UTF8 $outCsv

$modes = @(
    @{ mode = "avx16"; threads = 1 },
    @{ mode = "thread_flat"; threads = 8 },
    @{ mode = "openmp_flat"; threads = 8 },
    @{ mode = "ivf_pq"; threads = 1 },
    @{ mode = "thread_ivf_pq"; threads = 8 },
    @{ mode = "openmp_ivf_pq"; threads = 8 },
    @{ mode = "hnsw"; threads = 1 },
    @{ mode = "thread_hnsw"; threads = 8 },
    @{ mode = "openmp_hnsw"; threads = 8 }
)

foreach ($entry in $modes) {
    Write-Host "=== $($entry.mode) threads=$($entry.threads) ==="
    $output = & $exe $entry.mode $entry.threads 2>&1
    $output | ForEach-Object { Write-Host $_ }

    $recall = @($output | Select-String "average recall:" | ForEach-Object {
        ($_ -split ":\s+")[1]
    } | Select-Object -Last 1)[0]
    $latency = @($output | Select-String "average latency" | ForEach-Object {
        (($_ -split ":\s+")[1] -split "\s+")[0]
    } | Select-Object -Last 1)[0]
    $total = @($output | Select-String "total time" | ForEach-Object {
        (($_ -split ":\s+")[1] -split "\s+")[0]
    } | Select-Object -Last 1)[0]

    "$($entry.mode),$($entry.threads),$recall,$latency,$total" | Add-Content -Encoding UTF8 $outCsv
}

Write-Host ""
Write-Host "Saved to $outCsv"
