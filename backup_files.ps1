# Auto backup script for important files
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$files = @("execute_actions.py", "order_executor.py", "risk_manager.py", "unified_auto_trading_system.py")

foreach ($file in $files) {
    if (Test-Path $file) {
        Copy-Item $file "backups\${file}_$timestamp" -Force
        Write-Host "Backed up: $file"
    }
}

Write-Host "Backup completed at $timestamp"