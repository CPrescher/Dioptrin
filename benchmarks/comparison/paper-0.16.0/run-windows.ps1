<#
.SYNOPSIS
Dioptrin 0.16.0 paper rerun on the Windows workstation (RTX 5070 Ti, CUDA).

.DESCRIPTION
Protocol: docs/benchmarks/2026-11-25-paper-0.16.0/README.md

Start from any directory (the script changes to the checkout root itself):

  powershell -ExecutionPolicy Bypass -File benchmarks\comparison\paper-0.16.0\run-windows.ps1
  powershell -ExecutionPolicy Bypass -File benchmarks\comparison\paper-0.16.0\run-windows.ps1 -Quick
  powershell -ExecutionPolicy Bypass -File benchmarks\comparison\paper-0.16.0\run-windows.ps1 -DryRun

The passes run in order of importance. Each writes its own directory under
benchmark-results\ and its own <name>.log. A failed pass does not stop the next
one. A pass whose result directory already exists is skipped (compare.py
refuses to write into an existing directory), so after an interruption delete
the unfinished pass's directory and start the script again. Start and exit
lines with exit codes and time stamps go to the console and to
benchmark-results\run-windows-0.16.0.log.

Written for Windows PowerShell 5.1. Not run on Windows before it was committed.

.PARAMETER Quick
Run only the quick-suite installation check (a few minutes).

.PARAMETER DryRun
Print the command lines without running anything.
#>
[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$DryRun
)

Set-StrictMode -Version 2.0
# 'Stop' applies to this script's own cmdlets. The benchmark itself runs in
# cmd.exe (see Invoke-Pass), so its output never reaches PowerShell's error
# stream and its exit code is read from $LASTEXITCODE.
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location -LiteralPath $repo
[System.IO.Directory]::SetCurrentDirectory($repo)

# Relative paths on purpose: the command lines below then contain no spaces
# from the checkout location.
$py     = '.venv-benchmark\Scripts\python.exe'
$cmp    = 'benchmarks/comparison/compare.py'
$out    = 'benchmark-results'
$runLog = Join-Path $out 'run-windows-0.16.0.log'

$cpu   = @('dioptrin-pixel_split', 'dioptrin-supersampled', 'dioptrin-pyfai_full')
$cuda  = @('dioptrin-pixel_split-cuda', 'dioptrin-pyfai_full-cuda',
           'dioptrin-pixel_split-cuda-batch', 'dioptrin-pyfai_full-cuda-batch')
# Stock pyfai-full and pyfai-full-omp are left out of the 2D passes, as on
# 24 November (uninitialised scratch buffer in pyFAI 2026.9.0 full splitting).
$cpu1d = $cpu + @('azint', 'azint-cached', 'pyfai-full', 'pyfai-bbox', 'pyfai-full-omp')
$cpu2d = $cpu + @('azint', 'azint-cached', 'pyfai-bbox')
$all1d = $cpu1d + @('pyfai-full-opencl') + $cuda
$all2d = $cpu2d + @('pyfai-full-opencl') + $cuda
$gpu   = @('--opencl-device', 'gpu', '--cuda-device', '0', '--gpu-precision', 'f32_compensated')
$cakes = @('--dioptrin-2d-path', 'batch', '--dioptrin-2d-batch-split', 'nested')
$long  = @('--trials', '3', '--seconds', '1.0')

$script:failed = @()
$script:skipped = @()

function Get-Stamp {
    return (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
}

function Write-RunLog([string]$Line) {
    Write-Host $Line
    if (-not $DryRun) {
        Add-Content -LiteralPath $runLog -Value $Line -Encoding ASCII
    }
}

# Runs one compare.py pass.
#
# Why cmd.exe and a .cmd file instead of "& $py @argv *> $log": when Windows
# PowerShell 5.1 redirects the stderr of a native program, it wraps every
# stderr line in an ErrorRecord. Under $ErrorActionPreference = 'Stop' the
# first progress line on stderr then aborts the script, and under 'Continue'
# $? and the error stream still report failures for runs that succeeded; the
# log file is also written as UTF-16. Letting cmd.exe do the redirection
# (> log 2>&1) keeps the output as plain bytes, leaves PowerShell's error
# handling out of it, and gives the real exit code through "exit /b".
# The command line goes into benchmark-results\<name>.cmd, which also records
# exactly what ran. All arguments are fixed strings without spaces, quotes or
# percent signs, so quoting each one in double quotes is safe in cmd.exe.
function Invoke-Pass([string]$Name, [string[]]$PassArgs) {
    $argv = @('-E', '-s', '-B', $cmp, '--output', "$out/$Name") + $PassArgs
    $quoted = ($argv | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $logFile = "$out\$Name.log"
    $line = '"' + $py + '" ' + $quoted + ' > "' + $logFile + '" 2>&1'
    if ($DryRun) {
        Write-Host $line
        return
    }
    if (Test-Path -LiteralPath (Join-Path $out $Name)) {
        Write-RunLog "=== $Name skipped: $out\$Name already exists $(Get-Stamp)"
        $script:skipped += $Name
        return
    }
    $cmdFile = "$out\$Name.cmd"
    # %~dp0.. is the checkout root (the .cmd file lives in benchmark-results\).
    $content = @('@echo off', 'cd /d "%~dp0.."', $line, 'exit /b %ERRORLEVEL%')
    Set-Content -LiteralPath $cmdFile -Value $content -Encoding ASCII
    Write-RunLog "=== $Name start $(Get-Stamp)"
    & cmd.exe /d /c $cmdFile
    $code = $LASTEXITCODE
    Write-RunLog "=== $Name exit $code $(Get-Stamp)"
    if ($code -ne 0) {
        $script:failed += "$Name (exit $code)"
    }
}

if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    if ($DryRun) {
        Write-Host "note: $py not found; printing the commands only"
    } else {
        Write-Host "error: $py not found in $repo."
        Write-Host 'Build the environment first, in a Visual Studio Developer PowerShell:'
        Write-Host '  python benchmarks/comparison/setup.py --with-cuda --with-opencl'
        exit 1
    }
}
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $out | Out-Null
}

if ($Quick) {
    # Installation check: every variant of the passes below on the small
    # synthetic dataset at 1 and 2 workers. Stock pyfai-full / pyfai-full-omp
    # 2D rows at 2 workers may fail their self-consistency check (known pyFAI
    # bug); see the protocol README for what to look at.
    Invoke-Pass '2026-11-25-windows-quick' (@('--suite', 'quick', '--dimensions', '1d', '2d') + $cakes + $gpu + @('--variants') + $all1d)
} else {
    Write-RunLog "=== run-windows.ps1 start $(Get-Stamp)"
    # 1. Call-path ablation, Dioptrin CPU only (suite defaults, 1 trial)
    Invoke-Pass '2026-11-25-windows-ablation-paths' (@('--dimensions', '1d', '2d', '--dioptrin-1d-path', 'global', 'split', '--dioptrin-2d-path', 'sequential', '--variants') + $cpu)
    # 2. Primary float64 passes, 1 to 8 workers, 3 trials of 3 x 1 s
    Invoke-Pass '2026-11-25-windows-f64-1d' (@('--dimensions', '1d') + $long + $gpu + @('--variants') + $all1d)
    Invoke-Pass '2026-11-25-windows-f64-2d' (@('--dimensions', '2d') + $long + $cakes + $gpu + @('--variants') + $all2d)
    # 3. SMT extension, 12 and 16 workers, 64 frames per call, CPU variants
    Invoke-Pass '2026-11-25-windows-f64-1d-smt' (@('--dimensions', '1d', '--workers', '1,12,16', '--frames', '64', '--variants') + $cpu1d)
    Invoke-Pass '2026-11-25-windows-f64-2d-smt' (@('--dimensions', '2d', '--workers', '1,12,16', '--frames', '64') + $cakes + @('--variants') + $cpu2d)
    # 4. float32 inputs, 1 to 8 workers
    Invoke-Pass '2026-11-25-windows-f32-1d' (@('--dimensions', '1d', '--input-dtype', 'float32') + $gpu + @('--variants') + $all1d)
    Invoke-Pass '2026-11-25-windows-f32-2d' (@('--dimensions', '2d', '--input-dtype', 'float32') + $cakes + $gpu + @('--variants') + $all2d)
    # 5. CUDA float64 accumulation
    Invoke-Pass '2026-11-25-windows-cuda-f64' (@('--dimensions', '1d', '2d', '--workers', '1', '--cuda-device', '0', '--gpu-precision', 'f64') + $cakes + @('--variants') + $cuda)
}

if ($DryRun) {
    exit 0
}
if ($script:skipped.Count -gt 0) {
    Write-RunLog ("=== skipped (directory exists): " + ($script:skipped -join ', '))
}
if ($script:failed.Count -gt 0) {
    Write-RunLog ("=== failed: " + ($script:failed -join ', ') + " $(Get-Stamp)")
    exit 1
}
Write-RunLog "=== all done $(Get-Stamp)"
exit 0
