@echo off
title Project Golem - neural chat
ssh -tt specht32 "cd /home/ender/golem && ./golem neural chat --checkpoint runs/chat-live/champion.pt --temperature 0 --max-bytes 120 --context-bytes 768" <CONIN$ >CONOUT$ 2>CONOUT$
echo. >CONOUT$
echo Golem chat has closed. Press a key to close this window. >CONOUT$
pause <CONIN$ >CONOUT$
