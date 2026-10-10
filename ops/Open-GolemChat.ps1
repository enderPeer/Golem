param([string]$HostAlias = 'specht32')
$ErrorActionPreference = 'Stop'
if ($HostAlias -notin @('specht32', 'falke64', 'adler40', 'knecht24')) {
    throw 'Choose a configured cluster alias.'
}
$Host.UI.RawUI.WindowTitle = 'Project Golem - neural chat'
& ssh -t $HostAlias 'cd /home/ender/golem && ./golem neural chat --checkpoint runs/chat-live/champion.pt --temperature 0 --max-bytes 120 --context-bytes 768'
if ($LASTEXITCODE -ne 0) { Write-Host 'The remote chat exited. Check the saved checkpoint and connection.' }
