@echo off
title Project Golem - original English evolution
echo Project Golem - evolved genome 8cf3677f, selected generation 3136 >CONOUT$
echo Earlier English genetic model, without subsequent chat fine-tuning. >CONOUT$
echo Corpus-guided mutations and selection; original text includes some dialogue. >CONOUT$
echo Raw text continuation with at most 5 characters of effective context. >CONOUT$
echo Commands: /info /reset /reload /quit >CONOUT$
echo. >CONOUT$
ssh -tt -o BatchMode=yes -o ConnectTimeout=15 specht32 "cd /home/ender/golem && env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python3 -B -m english_world chat --checkpoint runs/chat-previews/raw-evolution-20261012T0842/champion.json --max-chars 160 --context-chars 2048 --temperature 0.8 --seed 1" <CONIN$ >CONOUT$ 2>CONOUT$
echo. >CONOUT$
echo Golem evolution chat has closed. Press a key to close this window. >CONOUT$
pause <CONIN$ >CONOUT$
