"""All living genes of a DNA-3 world, read from an engine checkpoint (docs/17 section 14), grouped by operation
sequence (synonymous codons collapsed) and by spelling.

  python tools/dna3_ckpt_genes.py RUN.ckpt --days 1,2,3,4,5,6 [--uniform-code] [--with OUT] [--top 30] [--json out.json]

The checkpoint (version 3) holds the packed forward strand and the live flags; the gene of a live slot is read from
codon 0 up to its first STOP with the codon table of the given days (the same tables as sim/dna_ref.py). --with OP
keeps only genes containing that operation (e.g. OUT for the taskers of day 6, EAT for eaters). Prints the most common
operation sequences with their counts and their most common spelling. Needs numpy; reads the whole strand (25 MB at
10^8 cells, 750 MB at 3 x 10^9).
"""
import argparse
import collections
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import dna_ref as R  # noqa: E402


def table_for(days, uniform):
    t = R.uniform_table() if uniform else R.standard_table()
    t = R.dna2_table(t, uniform)
    if 3 in days:
        t = R.dna3_table(t, uniform)
    if 4 in days:
        t = R.dna3_day4_table(t, uniform)
    if 5 in days:
        t = R.dna3_day5_table(t, uniform)
    if 6 in days:
        t = R.dna3_day6_table(t, uniform)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--days", default="1,2,3,4,5,6")
    ap.add_argument("--uniform-code", action="store_true")
    ap.add_argument("--with", dest="with_op", default=None)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    days = [int(x) for x in a.days.split(",")]
    table = table_for(days, a.uniform_code)
    with open(a.ckpt, "rb") as f:
        magic = f.read(8)
        ver, N, G, S, seed, flags, t_next = struct.unpack("<7Q", f.read(56))
        if magic != b"DNA3CKPT" or ver != 3:
            raise SystemExit("not a version-3 DNA-3 checkpoint")
        nbytes = (N + 3) // 4
        fwd_packed = np.frombuffer(f.read(nbytes), dtype=np.uint8)
        f.seek(nbytes, 1)                                   # rev
        live = np.frombuffer(f.read(S), dtype=np.uint8)
    maxc = G // 3
    slots = np.nonzero(live)[0]
    # unpack only the codon cells of the live slots
    cells = np.zeros((len(slots), 3 * maxc), dtype=np.uint8)
    idx = (slots[:, None] * G + np.arange(3 * maxc)[None, :])
    cells = (fwd_packed[idx >> 2] >> ((idx & 3) * 2).astype(np.uint8)) & 3
    codons = cells[:, 0::3] * 16 + cells[:, 1::3] * 4 + cells[:, 2::3]
    tab = np.array(table, dtype=np.int16)
    ops = tab[codons]
    by_ops, spell = collections.Counter(), collections.defaultdict(collections.Counter)
    want = R.OP_INDEX[a.with_op] if a.with_op else None
    for row_c, row_o in zip(codons, ops):
        stop = np.nonzero(row_o == R.STOP)[0]
        n = stop[0] if len(stop) else maxc
        o = tuple(int(x) for x in row_o[:n])
        if want is not None and want not in o[1:]:
            continue
        key = " ".join(R.OP_NAMES[x] for x in o)
        by_ops[key] += 1
        spell[key][" ".join(R.codon_text(int(c)) for c in row_c[:n])] += 1
    total = sum(by_ops.values())
    print("checkpoint at tick %d: %d live, %d selected genes, %d operation sequences" % (t_next, len(slots), total, len(by_ops)))
    out = []
    for key, cnt in by_ops.most_common(a.top):
        sp, sc = spell[key].most_common(1)[0]
        print("%7d  %-70s  %s (%d)" % (cnt, key, sp, sc))
        out.append({"count": cnt, "ops": key, "spelling": sp, "spelling_count": sc})
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"tick": t_next, "live": int(len(slots)), "selected": total, "sequences": len(by_ops), "top": out}, f, indent=1)


if __name__ == "__main__":
    main()
