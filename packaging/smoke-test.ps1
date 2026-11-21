# Runs a built wp-generator executable the way a user would, from an empty folder.
# Usage: pwsh packaging/smoke-test.ps1 <path to wp-generator executable>
param([Parameter(Mandatory)] [string] $Exe)
$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path $Exe).Path
$work = Join-Path ([IO.Path]::GetTempPath()) ("wp-smoke-" + [guid]::NewGuid())
New-Item -ItemType Directory $work | Out-Null
Push-Location $work

function Invoke-Wp {
    Write-Host "> wp-generator $args"
    & $Exe @args
    if ($LASTEXITCODE) { throw "wp-generator $args exited with $LASTEXITCODE" }
}

try {
    Invoke-Wp --version
    Invoke-Wp --preset nocturne --res 320 200 --seed 1 --output wall.png
    Invoke-Wp --metadata wall.png
    Invoke-Wp --preset cupertino --engine simplex --res 320 200 --seed 2 --output simplex.png
    Invoke-Wp --preset cupertino --engine spline-wave --res 320 200 --seed 3 --output spline.png
    Invoke-Wp --preset afterglow --res 160 100 --count 2 --workers 2 --seed 5 --out-dir batch
    $count = (Get-ChildItem batch -Filter *.png).Count
    if ($count -ne 2) { throw "expected 2 batch images, got $count" }
    Write-Host "smoke test passed"
}
finally {
    Pop-Location
}
