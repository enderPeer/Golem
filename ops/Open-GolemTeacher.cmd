@echo off
title Project Golem - teacher checkpoint preview
echo Project Golem - experimental teacher checkpoint 11b4a351 >CONOUT$
echo This model can give incorrect replies and invent names. >CONOUT$
echo Commands: /info /meaning /memory /reset /quit >CONOUT$
echo. >CONOUT$
ssh -tt -o BatchMode=yes -o ConnectTimeout=15 specht32 "cd /home/ender/golem && env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/home/ender/golem/cache nice -n 19 /home/ender/golem/.venv-neural/bin/python -B -m english_world neural world-chat --checkpoint runs/chat-previews/teacher-20261012T0640/checkpoint.pt --threads 1 --temperature 0 --max-bytes 120" <CONIN$ >CONOUT$ 2>CONOUT$
echo. >CONOUT$
echo Golem teacher chat has closed. Press a key to close this window. >CONOUT$
pause <CONIN$ >CONOUT$
