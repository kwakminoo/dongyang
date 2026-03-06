# 현재 열린 Java 파일을 컴파일 후 실행 (출력만 표시, ANSI 장식 없음)
param([string]$SrcFile)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$srcDir = Join-Path $root "src"

if (-not $SrcFile -or -not (Test-Path $SrcFile)) {
    Write-Host "Usage: run-current.ps1 -SrcFile <path-to-java-file>"
    exit 1
}

$fileNorm = $SrcFile -replace "\\", "/"
$fileNormLower = $fileNorm.ToLowerInvariant()
$rootNorm = ($root -replace "\\", "/").ToLowerInvariant()
if ($fileNormLower.StartsWith($rootNorm)) {
    $rel = $fileNorm.Substring($rootNorm.Length).TrimStart("/")
} else {
    $idx = $fileNormLower.IndexOf("/src/")
    if ($idx -lt 0) { $idx = [math]::Max($fileNormLower.IndexOf("\src\"), $fileNormLower.IndexOf("\src/")) }
    if ($idx -lt 0) { $rel = $fileNorm } else { $rel = "src/" + $fileNorm.Substring($idx + 5) }
}
$rel = $rel -replace "\\", "/"
if ($rel -notmatch "^src(/|\\).*\.java$") {
    Write-Host "File must be under src/ and end with .java"
    exit 1
}
$withoutSrc = $rel -replace "^src/", "" -replace "^src\\", ""
$className = $withoutSrc.Replace("/", ".").Replace(".java", "")

$binDir = Join-Path $root "bin"
if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir | Out-Null }

# 컴파일
& javac -encoding UTF-8 -d $binDir $SrcFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 한글 등 UTF-8 입력/출력을 위해 콘솔·JVM을 UTF-8로 설정 후 실행
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
& java "-Dfile.encoding=UTF-8" -cp $binDir $className
