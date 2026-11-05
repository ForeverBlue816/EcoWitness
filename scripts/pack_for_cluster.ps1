# Windows side: build the data (if needed) and pack it for Leonardo.
#   powershell -ExecutionPolicy Bypass -File scripts\pack_for_cluster.ps1            # data/*.jsonl + satellite_rgb/ (about 500 MB)
#   powershell -ExecutionPolicy Bypass -File scripts\pack_for_cluster.ps1 -DataOnly  # data/*.jsonl only (images already on the cluster)
# Output: ..\ecowitness_data.tar.gz or ..\ecowitness_jsonl.tar.gz, plus its sha256.
# Code is NOT packed: it is cloned from GitHub on the cluster.
# tar is used instead of Compress-Archive because Windows PowerShell 5.1 can write backslashes into zip entry names,
# which Linux unzip turns into literal file names such as "data\train.en.jsonl".
param([switch]$DataOnly)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)   # ecowitness/
$ws = Split-Path -Parent $here                                                 # folder that holds satellite_rgb/
$py = "$env:USERPROFILE\.local\bin\python3.14.exe"
Set-Location $here
if (-not (Test-Path "$here\data\train.en.jsonl")) {
  & $py -X utf8 -m ecowitness.build_revisions --splits train val test --out "$here\data"
}
$name = if ($DataOnly) { 'ecowitness_jsonl.tar.gz' } else { 'ecowitness_data.tar.gz' }
$out = "$ws\$name"
if (Test-Path $out) { Remove-Item $out }
# Relative paths only: Windows PowerShell 5.1 mangles non-ASCII folder names (e.g. 数据集) passed to native tar.exe.
$code = Split-Path -Leaf $here
Set-Location $ws
if ($DataOnly) {
  & "$env:SystemRoot\System32\tar.exe" -czf $name -C $code data
} else {
  & "$env:SystemRoot\System32\tar.exe" -czf $name -C $code data -C .. satellite_rgb
}
if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
$hash = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
"$hash  $name" | Out-File -Encoding ascii "$out.sha256"
Write-Host "data: $out  sha256 $hash"
Write-Host "upload:   scp $out ychen003@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ychen003/"
if ($DataOnly) {
  Write-Host 'replace:  cd $SCRATCH/ecowitness && mv data data_old && tar -xzf $SCRATCH/ecowitness_jsonl.tar.gz -C $SCRATCH/ecowitness'
} else {
  Write-Host 'unpack:   mkdir -p $SCRATCH/ecowitness && tar -xzf $SCRATCH/ecowitness_data.tar.gz -C $SCRATCH/ecowitness'
}
