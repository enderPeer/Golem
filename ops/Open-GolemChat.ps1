$ErrorActionPreference = 'Stop'
$golemChatScript = Join-Path $PSScriptRoot 'Open-GolemChat.cmd'
# CONIN$/CONOUT$ in the CMD wrapper ensure the new window has its own keyboard
# and display even when its launcher inherited redirected background-tool pipes.
Start-Process -FilePath cmd.exe -ArgumentList @('/d', '/c', ('"' + $golemChatScript + '"')) -WindowStyle Normal
