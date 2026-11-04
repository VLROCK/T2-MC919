param(
    [string]$Images = "data/images",
    [string]$Work = "work/main",
    [ValidateSet("sequential", "exhaustive")][string]$Matching = "sequential",
    [string]$Colmap = "colmap",
    [double]$MarkerSizeM = 0,
    [int]$MarkerId = 0,
    [int]$MaxImages = 0,
    [switch]$CpuOnly,
    [switch]$Dense,
    [switch]$Mesh,
    [switch]$LoopDetection
)
$ErrorActionPreference = "Stop"
function Invoke-Python([string[]]$Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python falhou com código $LASTEXITCODE" }
}
$pipeline = @("-m", "src.pipeline", "--images", $Images, "--work", $Work,
              "--matching", $Matching, "--colmap", $Colmap)
if ($CpuOnly) { $pipeline += "--cpu" }
if ($Dense) { $pipeline += "--dense" }
if ($Mesh) { $pipeline += "--mesh" }
if ($LoopDetection) { $pipeline += "--loop-detection" }
if ($MaxImages -gt 0) { $pipeline += @("--max-images", "$MaxImages") }
Invoke-Python $pipeline

$modelImages = $Images
if (Test-Path -LiteralPath (Join-Path $Work "images")) { $modelImages = Join-Path $Work "images" }
$scaleArgs = @()
if ($MarkerSizeM -gt 0) {
    $markerSizeInvariant = $MarkerSizeM.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    Invoke-Python @("-m", "src.scale_marker", "--model", (Join-Path $Work "sparse_txt"),
                    "--images", $modelImages, "--marker-size-m", $markerSizeInvariant,
                    "--marker-id", "$MarkerId", "--output", (Join-Path $Work "scale.json"))
    $scaleArgs = @("--scale-json", (Join-Path $Work "scale.json"))
}
$featureArgs = @()
$intrinsics = Join-Path $Work "intrinsics.json"
if (Test-Path -LiteralPath $intrinsics) { $featureArgs = @("--intrinsics", $intrinsics) }
Invoke-Python (@("-m", "src.feature_analysis", "--images", $modelImages,
                 "--output", (Join-Path $Work "features")) + $featureArgs)
Invoke-Python (@("-m", "src.analysis", "--model", (Join-Path $Work "sparse_txt"),
                 "--manifest", (Join-Path $Work "manifest.json"),
                 "--measurements", "data/measurements.json",
                 "--output", (Join-Path $Work "analysis")) + $scaleArgs)
Invoke-Python @("-m", "src.visualize", "--model", (Join-Path $Work "sparse_txt"),
                "--metrics", (Join-Path $Work "analysis/metrics.json"),
                "--output", (Join-Path $Work "analysis/navigation.mp4"))
Write-Host "Resultados em $Work"
