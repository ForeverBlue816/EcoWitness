# Windows side: build the data (if needed) and pack it for Leonardo.
#   powershell -ExecutionPolicy Bypass -File scripts\pack_for_cluster.ps1
# Output: ..\ecowitness_data.tar.gz (data/*.jsonl + satellite_rgb/, about 500 MB) and its sha256.
# Code is NOT packed: it is cloned from GitHub on the cluster.
# tar is used instead of Compress-Archive because Windows PowerShell 5.1 can write backslashes into zip entry names,
# which Linux unzip turns into literal file names such as "data\train.en.jsonl".
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)   # ecowitness/
$ws = Split-Path -Parent $here                                                 # folder that holds satellite_rgb/
$py = "$env:USERPROFILE\.local\bin\python3.14.exe"
Set-Location $here
if (-not (Test-Path "$here\data\train.en.jsonl")) {
  & $py -X utf8 -m ecowitness.build_revisions --splits train val test --out "$here\data"
}
$out = "$ws\ecowitness_data.tar.gz"
if (Test-Path $out) { Remove-Item $out }
& "$env:SystemRoot\System32\tar.exe" -czf $out -C $here data -C $ws satellite_rgb
$hash = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
"$hash  ecowitness_data.tar.gz" | Out-File -Encoding ascii "$out.sha256"
Write-Host "data: $out  sha256 $hash"
Write-Host "upload:   scp $out ychen003@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ychen003/"
Write-Host 'unpack:   mkdir -p $SCRATCH/ecowitness && tar -xzf $SCRATCH/ecowitness_data.tar.gz -C $SCRATCH/ecowitness'
