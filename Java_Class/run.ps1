# 터미널에서 Java 실행 (인자 없으면 ch01/HelloWorld 실행)
# 사용: .\run.ps1   또는   .\run.ps1 src\ch01\HelloWorld.java
param([string]$SrcFile)

$root = $PSScriptRoot
if (-not $SrcFile) {
    $SrcFile = Join-Path $root "src\ch01\HelloWorld.java"
} elseif (-not [System.IO.Path]::IsPathRooted($SrcFile)) {
    $SrcFile = Join-Path $root $SrcFile
}

& "$root\run-current.ps1" -SrcFile $SrcFile
