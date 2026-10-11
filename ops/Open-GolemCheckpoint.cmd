@echo off
title Project Golem - new checkpoint preview
echo Project Golem - experimental checkpoint 29814478 >CONOUT$
echo This newer model still makes mistakes and invents names. >CONOUT$
echo Commands: /info /meaning /memory /reset /quit >CONOUT$
echo. >CONOUT$
ssh -tt specht32 "cd /home/ender/golem && env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/home/ender/golem/cache nice -n 19 /home/ender/golem/.venv-neural/bin/python -B -m english_world neural world-chat --checkpoint runs/chat-previews/trajectory-20261011T1008/checkpoint.pt --threads 1 --temperature 0 --max-bytes 120" <CONIN$ >CONOUT$ 2>CONOUT$
echo. >CONOUT$
echo Golem checkpoint chat has closed. Press a key to close this window. >CONOUT$
pause <CONIN$ >CONOUT$
