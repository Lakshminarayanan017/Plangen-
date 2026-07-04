Add-Type -Assembly 'System.IO.Compression.FileSystem'
$zip = [System.IO.Compression.ZipFile]::OpenRead('c:\Users\Welcome\Desktop\plangen.zip')
$entries = $zip.Entries | Select-Object -ExpandProperty FullName | Sort-Object
$zip.Dispose()

Write-Host "=== Total entries in zip: $($entries.Count) ==="
Write-Host ""
Write-Host "=== First 10 entries (shows root structure): ==="
$entries | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "=== Entries containing 'training': ==="
$entries | Where-Object { $_ -match 'training' } | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "=== Entries containing 'model_trainer': ==="
$entries | Where-Object { $_ -match 'model_trainer' } | ForEach-Object { Write-Host "  $_" }
