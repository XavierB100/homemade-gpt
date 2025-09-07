# Create desktop shortcut for HomeMade GPT with custom AI icon

$currentDir = (Get-Location).Path
$batchPath = Join-Path $currentDir "HomeMade GPT.bat"
$iconPath = Join-Path $currentDir "homemade_gpt.ico"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "HomeMade GPT.lnk"

Write-Host "Creating desktop shortcut with custom AI icon..." -ForegroundColor Green
Write-Host "Batch file: $batchPath" -ForegroundColor Cyan
Write-Host "Icon file: $iconPath" -ForegroundColor Cyan
Write-Host "Shortcut location: $shortcutPath" -ForegroundColor Cyan

# Check if batch file exists
if (-not (Test-Path $batchPath)) {
    Write-Host "ERROR: Batch file not found!" -ForegroundColor Red
    exit 1
}

# Check if icon exists
if (-not (Test-Path $iconPath)) {
    Write-Host "WARNING: Custom icon not found, using default" -ForegroundColor Yellow
    $iconPath = $null
}

# Remove old shortcut if it exists
if (Test-Path $shortcutPath) {
    Remove-Item $shortcutPath -Force
    Write-Host "Removed existing shortcut" -ForegroundColor Yellow
}

# Create shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $batchPath
$Shortcut.WorkingDirectory = $currentDir
$Shortcut.Description = "HomeMade GPT - AI Training Platform"

# Set custom icon if available
if ($iconPath) {
    $Shortcut.IconLocation = $iconPath
    Write-Host "Applied custom AI-themed icon" -ForegroundColor Green
}

$Shortcut.Save()

Write-Host "✅ Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "🎨 Your shortcut now has a professional AI-themed icon!" -ForegroundColor Magenta
Write-Host "Click the 'HomeMade GPT' icon on your desktop to launch!" -ForegroundColor Yellow
