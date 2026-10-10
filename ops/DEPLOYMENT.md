Run these commands from the dedicated Linux checkout:

```bash
bash ops/preflight.sh
bash ops/run_pilot.sh --dry-run configs/pilot.toml data/english/dataset.json runs/pilot
bash ops/run_pilot.sh configs/pilot.toml data/english/dataset.json runs/pilot
```

The launcher resolves its project directory from its own path. Config, dataset
and output paths must stay inside that project; output must be new or empty.
It starts one uniquely named transient user systemd unit and prints the exact
status, log and stop commands for that unit. It does not install services,
enable autostart, change existing services or manage other jobs.

Limits: 400% CPU quota (four cores' worth), 8 GiB memory, 64 tasks, nice 19 and
1,800 seconds runtime. Numerical-library thread counts are four. The worker is
CPU-only, with CUDA/HIP/ROCR device visibility empty. Preflight reads Linux PCI
metadata without invoking a GPU driver utility or creating a GPU context.

The project must remain below 20 GiB and its filesystem must have more than
50 GiB free. Both are checked before launch and every five seconds while the
worker runs. This is a monitored stop policy, not a filesystem quota: writes
between checks can exceed a threshold briefly. On failure the guard stops only
the child process group it launched. systemd also bounds the entire unit.

The standalone preflight is read-only. Dry-run validates files and storage and
prints a shell-escaped command without starting any job. Linux/POSIX and a
working `systemd-run --user` session are required for the actual launch.

The frozen reference retains its original source bytes and license under
`upstream/dna2/`. Its nested `.gitattributes` disables line-ending conversion so
the SHA-256 provenance checks remain valid after a Linux checkout.
