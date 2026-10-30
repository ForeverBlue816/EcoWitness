# Windows side: build the data (if needed) and make two zips:
#   ecowitness_code.zip  (code, configs, scripts, README)       -> $HOME/ecowitness on Leonardo
#   ecowitness_data.zip  (data/*.jsonl + satellite_rgb/, ~1 GB) -> $SCRATCH/ecowitness on Leonardo
#   powershell -ExecutionPolicy Bypass -File scripts\pack_for_cluster.ps1
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)   # ecowitness/
$ws = Split-Path -Parent $here                                                 # 数据集/
$py = "$env:USERPROFILE\.local\bin\python3.14.exe"
Set-Location $here
if (-not (Test-Path "$here\data\train.en.jsonl")) {
  & $py -X utf8 -m ecowitness.build_revisions --splits train val test --out "$here\data"
}
Get-ChildItem -Recurse $here -Include __pycache__ -Directory | Remove-Item -Recurse -Force

$codeZip = "$ws\ecowitness_code.zip"
if (Test-Path $codeZip) { Remove-Item $codeZip }
Compress-Archive -Path "$here\ecowitness", "$here\scripts", "$here\configs", "$here\README.md", "$here\requirements.txt" -DestinationPath $codeZip

$stage = "$ws\_ew_data_stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -Recurse "$here\data" "$stage\data"
Copy-Item -Recurse "$ws\satellite_rgb" "$stage\satellite_rgb"
$dataZip = "$ws\ecowitness_data.zip"
if (Test-Path $dataZip) { Remove-Item $dataZip }
Compress-Archive -Path "$stage\*" -DestinationPath $dataZip
Remove-Item -Recurse -Force $stage

Write-Host "code: $codeZip"
Write-Host "data: $dataZip"
Write-Host "upload:"
Write-Host "  scp $codeZip ychen003@login.leonardo.cineca.it:~/"
Write-Host "  scp $dataZip ychen003@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ychen003/"
Write-Host "on Leonardo:"
Write-Host '  cd ~ && rm -rf ecowitness && mkdir ecowitness && unzip -q ~/ecowitness_code.zip -d ecowitness'
Write-Host '  mkdir -p $SCRATCH/ecowitness && unzip -q $SCRATCH/ecowitness_data.zip -d $SCRATCH/ecowitness'
