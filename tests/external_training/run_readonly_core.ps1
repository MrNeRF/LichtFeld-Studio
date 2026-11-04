# SPDX-License-Identifier: GPL-3.0-or-later
# Run from an x64 Visual Studio developer PowerShell, after building LichtFeld.
param(
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [string]$VcpkgPrefix,
    [string]$CudaDirectory=$env:CUDA_PATH,
    [ValidateSet('read_only_scene_core.cpp','splat_preview_core.cpp')]
    [string]$Source='read_only_scene_core.cpp'
)
$ErrorActionPreference='Stop'
$taskSource=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskBuild=(Resolve-Path -LiteralPath $BuildDirectory).Path
if(-not $VcpkgPrefix){$VcpkgPrefix=Join-Path $taskBuild 'vcpkg_installed/x64-windows'}
if(-not $CudaDirectory){throw 'Set CUDA_PATH or pass -CudaDirectory'}
$taskOutput=Join-Path $taskBuild 'readonly-tests'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskArguments=@('/nologo','/std:c++latest','/EHsc','/MD','/DNOMINMAX',
    '/DGLM_FORCE_DEPTH_ZERO_TO_ONE','/DFMT_SHARED',
    '/DLFS_ALLOCATION_PROFILING_ENABLED=0','/DLFS_CUDA_FAILURE_INJECTION_ENABLED=0')
foreach($taskInclude in @('src','src/core/include','src/diagnostics/include','include')){
    $taskArguments+=('/I'+(Join-Path $taskSource $taskInclude))
}
foreach($taskInclude in @((Join-Path $taskBuild 'include'),
    (Join-Path $taskBuild 'include/core-abi/Release'),
    (Join-Path $VcpkgPrefix 'include'),(Join-Path $CudaDirectory 'include'))){
    $taskArguments+=('/I'+$taskInclude)
}
$taskName=[System.IO.Path]::GetFileNameWithoutExtension($Source)
$taskExecutable=Join-Path $taskOutput ($taskName+'.exe')
$taskArguments+=@((Join-Path $PSScriptRoot $Source),
    ('/Fo'+(Join-Path $taskOutput ($taskName+'.obj'))),('/Fe'+$taskExecutable),
    '/link',(Join-Path $taskBuild 'src/core/lfs_core.lib'),(Join-Path $VcpkgPrefix 'lib/fmt.lib'))
& cl @taskArguments
if($LASTEXITCODE -ne 0){throw 'Core contract test compilation failed'}
$taskOriginalPath=$env:PATH
try{
    $env:PATH="$taskBuild;$(Join-Path $VcpkgPrefix 'bin');$(Join-Path $CudaDirectory 'bin');$env:PATH"
    & $taskExecutable
    if($LASTEXITCODE -ne 0){throw "Core contract test failed: $LASTEXITCODE"}
} finally {$env:PATH=$taskOriginalPath}
