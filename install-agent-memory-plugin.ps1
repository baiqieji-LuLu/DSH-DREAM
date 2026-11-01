#requires -Version 5.1
<#
.SYNOPSIS
  Install or refresh the DSH-DREAM DSH bundle in the desktop profile.

.DESCRIPTION
  Stops DeepSeek Harness, removes the obsolete hand-written `agent-memory`
  patch entry, links the plugin into the desktop profile, sets the shared
  bridge environment variables, restarts DSH, and waits for the Host to write
  the bridge JSON.

  Run from an elevated or normal PowerShell window; no administrator rights
  are required for the directory link created by `dsh plugin`.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\install-agent-memory-plugin.ps1 -Force

.EXAMPLE
  .\install-agent-memory-plugin.ps1 -PluginDir "G:\dsh_plugin_test_workspace\dsh_plugin" -Force
#>
[CmdletBinding()]
param(
    [string]$PluginDir = 'G:\dsh_plugin_test_workspace\dsh_plugin',
    [string]$ProjectRoot = 'G:\deepseek_ragproject_workspace',
    [string]$DshRoot = 'G:\deepseekharness',
    [string]$Profile = 'desktop',
    [switch]$Force,
    [switch]$NoRestart
)

$ErrorActionPreference = 'Stop'

$profileDir = Join-Path $env:USERPROFILE ".dsh\profiles\$Profile"
$profilePackage = Join-Path $profileDir 'package.json'
$profilePatch = Join-Path $profileDir 'cordis.patch.yml'
$dshCmd = Join-Path $DshRoot 'resources\runtime\cli\bin\dsh.cmd'
$dshExe = Join-Path $DshRoot 'DeepSeek Harness.exe'
$configPath = Join-Path $ProjectRoot 'agent_memory_config.json'
$ragPath = Join-Path $ProjectRoot '.chroma'
$bundleName = '@local/dsh-dream-settings'

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "    OK: $Message" -ForegroundColor Green
}

function Stop-Dsh {
    $running = @(Get-Process -Name 'DeepSeek Harness' -ErrorAction SilentlyContinue)
    if ($running.Count -eq 0) {
        Write-Ok 'DSH is not running.'
        return
    }

    if (-not $Force) {
        $answer = Read-Host "DSH is running ($($running.Count) processes). Stop it now? [y/N]"
        if ($answer -notmatch '^[yY]') {
            throw 'Cancelled: quit DSH completely before installing.'
        }
    }

    Write-Host "    Stopping $($running.Count) DSH process(es)..." -ForegroundColor Yellow
    $running | Stop-Process -Force
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Process -Name 'DeepSeek Harness' -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 250
    }
    if (Get-Process -Name 'DeepSeek Harness' -ErrorAction SilentlyContinue) {
        throw 'DSH did not stop within 15 seconds. Quit it from the tray and retry.'
    }
    Write-Ok 'DSH stopped.'
}

function Remove-ObsoletePatchEntry {
    if (-not (Test-Path -LiteralPath $profilePatch)) {
        Write-Ok 'No profile cordis.patch.yml; nothing to clean.'
        return
    }

    $backup = "$profilePatch.bak-$(Get-Date -Format 'yyyyMMddHHmmss')"
    Copy-Item -LiteralPath $profilePatch -Destination $backup -Force
    Write-Host "    Patch backup: $backup" -ForegroundColor DarkGray

    $lines = @(Get-Content -LiteralPath $profilePatch)
    $result = New-Object System.Collections.Generic.List[string]
    $skipping = $false
    $removed = 0

    foreach ($line in $lines) {
        if ($line -match '^\s*-\s*id:\s*agent-memory\s*$') {
            $skipping = $true
            $removed++
            continue
        }
        if ($skipping) {
            if ($line -match '^\s*-\s*') {
                $skipping = $false
            }
            else {
                continue
            }
        }
        $result.Add($line)
    }

    if ($removed -eq 0) {
        Write-Ok 'Obsolete agent-memory patch entry not present.'
        Remove-Item -LiteralPath $backup -Force
        return
    }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($profilePatch, $result, $encoding)
    Write-Ok "Removed $removed obsolete patch entry (the bundle patch inserts the real row)."
}

function Ensure-ProfileBundle {
    $manifest = Get-Content -LiteralPath $profilePackage -Raw | ConvertFrom-Json
    $changed = $false

    if (-not $manifest.dependencies) {
        $manifest | Add-Member -NotePropertyName dependencies -NotePropertyValue ([pscustomobject]@{}) -Force
        $changed = $true
    }
    $dependency = $manifest.dependencies.PSObject.Properties[$bundleName]
    if (-not $dependency) {
        $manifest.dependencies | Add-Member -NotePropertyName $bundleName -NotePropertyValue "link:$PluginDir" -Force
        $changed = $true
    }

    if (-not $manifest.dsh) {
        $manifest | Add-Member -NotePropertyName dsh -NotePropertyValue ([pscustomobject]@{}) -Force
        $changed = $true
    }
    if (-not $manifest.dsh.profile) {
        $manifest.dsh | Add-Member -NotePropertyName profile -NotePropertyValue ([pscustomobject]@{}) -Force
        $changed = $true
    }
    $bundles = @($manifest.dsh.profile.bundles)
    if ($bundles -notcontains $bundleName) {
        $manifest.dsh.profile | Add-Member -NotePropertyName bundles -NotePropertyValue @($bundles + $bundleName) -Force
        $changed = $true
    }

    if ($changed) {
        $json = $manifest | ConvertTo-Json -Depth 20
        $encoding = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($profilePackage, "$json`r`n", $encoding)
        Write-Ok 'Profile dependency and dsh.profile.bundles updated.'
    }
    else {
        Write-Ok 'Profile dependency and bundle selection already present.'
    }
}

Write-Step 'Preflight'
if (-not (Test-Path -LiteralPath $PluginDir)) { throw "Plugin directory not found: $PluginDir" }
if (-not (Test-Path -LiteralPath (Join-Path $PluginDir 'package.json'))) { throw "package.json not found in $PluginDir" }
if (-not (Test-Path -LiteralPath (Join-Path $PluginDir 'client.js'))) { throw "client.js not found in $PluginDir" }
$pluginManifest = Get-Content -LiteralPath (Join-Path $PluginDir 'package.json') -Raw | ConvertFrom-Json
if (-not $pluginManifest.dsh.client) { throw "dsh.client is missing from $PluginDir\package.json" }
if (-not $pluginManifest.exports.'./client') { throw "./client export is missing from $PluginDir\package.json" }
if (-not (Test-Path -LiteralPath $dshCmd)) { throw "dsh.cmd not found: $dshCmd" }
if (-not (Test-Path -LiteralPath $dshExe)) { throw "DSH executable not found: $dshExe" }
if (-not (Test-Path -LiteralPath $profilePackage)) { throw "Desktop profile not initialized: $profilePackage" }
Write-Ok "Plugin: $PluginDir"
Write-Ok "Profile: $profileDir"

Write-Step 'Stopping DSH'
Stop-Dsh

Write-Step 'Cleaning obsolete profile patch entry'
Remove-ObsoletePatchEntry

Write-Step 'Setting bridge environment variables'
New-Item -ItemType Directory -Force -Path $ProjectRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $configPath) | Out-Null
[Environment]::SetEnvironmentVariable('AGENT_MEMORY_CONFIG', $configPath, 'User')
[Environment]::SetEnvironmentVariable('AGENT_MEMORY_RAG_PATH', $ragPath, 'User')
$env:AGENT_MEMORY_CONFIG = $configPath
$env:AGENT_MEMORY_RAG_PATH = $ragPath
Write-Ok "AGENT_MEMORY_CONFIG=$configPath"
Write-Ok "AGENT_MEMORY_RAG_PATH=$ragPath"

if (-not (Test-Path -LiteralPath $configPath)) {
    $seed = [ordered]@{
        enable_memory           = $true
        rag_path                = $ragPath
        max_window_tokens       = 1500
        retrieval_top_k         = 5
        embedding_provider      = 'local'
        embedding_model         = ''
        embedding_api_key       = ''
        embedding_base_url      = ''
        auto_rebuild_collection = $true
    } | ConvertTo-Json
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($configPath, "$seed`r`n", $encoding)
    Write-Ok "Seeded initial bridge file: $configPath"
}

Write-Step 'Installing the plugin bundle'
Push-Location -LiteralPath $ProjectRoot
try {
    & $dshCmd plugin --profile $Profile add "link:$PluginDir"
    if ($LASTEXITCODE -ne 0) {
        throw "dsh plugin add failed with exit code $LASTEXITCODE. See $profileDir\.plugin-manager\logs."
    }
}
finally {
    Pop-Location
}
Write-Ok 'dsh plugin add completed.'

Write-Step 'Verifying profile manifest'
Ensure-ProfileBundle
$manifest = Get-Content -LiteralPath $profilePackage -Raw | ConvertFrom-Json
$dependency = $manifest.dependencies.PSObject.Properties[$bundleName]
if (-not $dependency) { throw "Dependency $bundleName is missing after install." }
if (@($manifest.dsh.profile.bundles) -notcontains $bundleName) { throw "Bundle $bundleName is missing from dsh.profile.bundles." }
Write-Ok "Dependency: $($dependency.Value)"
Write-Ok 'Bundle selection: present'

Write-Step 'Restarting DSH'
if ($NoRestart) {
    Write-Host '    -NoRestart set; start DSH manually.' -ForegroundColor Yellow
}
else {
    $beforeWrite = $null
    if (Test-Path -LiteralPath $configPath) {
        $beforeWrite = (Get-Item -LiteralPath $configPath).LastWriteTimeUtc
    }
    Start-Process -FilePath $dshExe -WorkingDirectory $DshRoot
    Write-Ok 'DSH started.'

    Write-Step 'Waiting for the Host to write the bridge file'
    $deadline = (Get-Date).AddSeconds(60)
    $written = $false
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $configPath) {
            $item = Get-Item -LiteralPath $configPath
            if ($null -eq $beforeWrite -or $item.LastWriteTimeUtc -gt $beforeWrite) {
                $written = $true
                break
            }
        }
        Start-Sleep -Milliseconds 500
    }
    if ($written) {
        Write-Ok "Bridge rewritten: $configPath"
        $bridge = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        if ($bridge.PSObject.Properties.Name -contains 'embedding_api_key') {
            $bridge.embedding_api_key = '***'
        }
        $bridge | ConvertTo-Json
    }
    else {
        Write-Warning "Bridge file was not rewritten within 60s. Check Settings > DSH-DREAM and the DSH logs."
    }
}

Write-Host "`nDone. Open DSH Settings and select 'DSH-DREAM'." -ForegroundColor Green
