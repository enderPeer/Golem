"""DNA-1 reference simulator: the Python side of the contract in docs/09_dna1_spec.md.

Pure Python 3.11, standard library only, deterministic. It implements every rule of docs/09 sections 1 to 11
(symbols and complement, packed 2-bit layout for hash and dump, slots and genes, the standard-code codon table
and the uniform code, the processor with its 21 operations and the snapshot rule, the eight-step tick, the
initial state, the splitmix64 PRNG, the gene and state hashes) as literally as possible; every place where the
specification left a choice follows docs/09 section 13 ("Clarifications") and is marked `# [13: title]` here.

The GPU engine (gpu/dna_world.cu) must print the same "tick <n> hash <hex>" lines on the same arguments.

  python sim/dna_ref.py --N 16384 --G 256 --ticks 200 --seed 1 --hash-every 50
  python sim/dna_ref.py --N 65536 --G 256 --ticks 300 --seed 3 --mu 0 --n-seed 4 \
      --genome "ATG GCT CCT ACT CTT GTT GGT TAA" --births births_t3.tsv --out census_t3.json

stdout: one line "tick <t> hash <16 lowercase hex>" per hashed state, where t = 0 .. ticks-1 labels the state
after tick t (the tick that used t in rnd); the initial state of section 6 is printed as "tick -1" only with
--hash-init [13: Tick labels]. --hash-every k > 0 hashes after every k-th tick ((t + 1) mod k = 0) and after the
last tick, 0 = after the last tick only [13: Hash schedule]. One line "census {json}" per census state, after
every census_every-th tick and after the last [13: Census schedule]. stderr: parameters at the start, a one-line
summary at the end.

DNA-2 (docs/15) is behind --dna2; DNA-3 (docs/17: day 1 light and energy, day 2 grid and firmament, day 3 terrain,
HARVEST, copy cost and maintenance) behind --days 1,2,3 or --dna3, each day implying --dna2:

  python sim/dna_ref.py --dna3 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 3 --n-seed 16 \
      --seed-terrain sea --genome "ATG AGA GCT CTT TTA GGT"

Honest caveats: this is a reference, not an engine. The tape is one byte per cell here (the packed layout is
only produced for the hash and the dump), so N = 3 x 10^9 does not fit; the test worlds of docs/09 section 10
(N up to 1,048,576) run in seconds to minutes. The whole state is rebuilt and FNV-hashed on every hashed tick,
which dominates the time at large N; use --hash-every 0 if only the final hash is needed.
"""
import argparse
import json
import struct
import sys
import time

MASK64 = (1 << 64) - 1
C1 = 0x9E3779B97F4A7C15   # splitmix64 increment, also the t-weight of mix()
C2 = 0xBF58476D1CE4E5B9   # first splitmix multiplier, also the k-weight
C3 = 0x94D049BB133111EB   # second splitmix multiplier, also the j-weight
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211

LETTER = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
LETTERS = "ACGT"
COMP_TABLE = bytes((b ^ 3) if b < 4 else b for b in range(256))   # comp(s) = s xor 3 on a byte-per-cell tape

# operation codes (the order is irrelevant; MEANINGS below carries the table order of docs/09 section 3)
NOP, START, STOP, COPY, R1, L1, JNZB, SETP, SETQ, ALLOC, DIVIDE, SKNZ, SKZ, LDA, STA, INC, FLIP, R3, L3, SEEK, \
    COPYR, REWIND, TERM, HARVEST, SENSE, WAIT, EAT, MOVE, EXEC, SIGNAL, LISTEN, IN, OUT = range(33)  # TERM only under --dna2 (docs/15), HARVEST day 3,
                                                                         # SENSE/WAIT day 4, EAT/MOVE/EXEC day 5, SIGNAL/LISTEN/IN/OUT day 6 (docs/17)
OP_NAMES = ["NOP", "START", "STOP", "COPY", "R1", "L1", "JNZB", "SETP", "SETQ", "ALLOC", "DIVIDE", "SKNZ", "SKZ",
            "LDA", "STA", "INC", "FLIP", "R3", "L3", "SEEK", "COPYR", "REWIND", "TERM", "HARVEST", "SENSE", "WAIT", "EAT", "MOVE",
            "EXEC", "SIGNAL", "LISTEN", "IN", "OUT"]
OP_INDEX = {n: i for i, n in enumerate(OP_NAMES)}

# docs/09 section 3, row by row (codons, amino acid, operation). The row order is the "table order" of the
# uniform code: codon c < 21 means row c.
CODON_ROWS = [
    ("ATG", "Met", "START"),
    ("TAA TAG TGA", "stop", "STOP"),
    ("TTA TTG CTT CTC CTA CTG", "Leu", "COPY"),
    ("TCT TCC TCA TCG AGT AGC", "Ser", "R1"),
    ("CGT CGC CGA CGG AGA AGG", "Arg", "L1"),
    ("GTT GTC GTA GTG", "Val", "JNZB"),
    ("CCT CCC CCA CCG", "Pro", "SETP"),
    ("ACT ACC ACA ACG", "Thr", "SETQ"),
    ("GCT GCC GCA GCG", "Ala", "ALLOC"),
    ("GGT GGC GGA GGG", "Gly", "DIVIDE"),
    ("ATT ATC ATA", "Ile", "SKNZ"),
    ("TTT TTC", "Phe", "SKZ"),
    ("TAT TAC", "Tyr", "LDA"),
    ("CAT CAC", "His", "STA"),
    ("CAA CAG", "Gln", "INC"),
    ("AAT AAC", "Asn", "FLIP"),
    ("AAA AAG", "Lys", "R3"),
    ("GAT GAC", "Asp", "L3"),
    ("GAA GAG", "Glu", "SEEK"),
    ("TGT TGC", "Cys", "COPYR"),
    ("TGG", "Trp", "REWIND"),
]
MEANINGS = [OP_INDEX[row[2]] for row in CODON_ROWS]   # 21 meanings in table order


def codon_value(text):
    """'ATG' -> 16*A + 4*T + G = 14. Accepts U for T and lowercase."""
    text = text.upper()
    if len(text) != 3 or any(ch not in LETTER for ch in text):
        raise ValueError("codon must be three letters of ACGT(U): %r" % text)
    return 16 * LETTER[text[0]] + 4 * LETTER[text[1]] + LETTER[text[2]]


def codon_text(c):
    return LETTERS[c >> 4] + LETTERS[(c >> 2) & 3] + LETTERS[c & 3]


def standard_table():
    """codon value 0..63 -> operation code, from the rows of section 3 (checked to cover all 64 exactly once)."""
    table = [None] * 64
    for codons, _aa, name in CODON_ROWS:
        for cod in codons.split():
            v = codon_value(cod)
            if table[v] is not None:
                raise AssertionError("codon %s listed twice" % cod)
            table[v] = OP_INDEX[name]
    if any(op is None for op in table):
        raise AssertionError("codon table does not cover all 64 codons")
    return table


def uniform_table():
    """--uniform-code: codons 0..20 are the 21 meanings in table order, 21..63 are NOP. [13: START under the
    uniform code: codon 0 = AAA is START, codon 1 = AAC is STOP]"""
    return [MEANINGS[c] if c < len(MEANINGS) else NOP for c in range(64)]


DNA2_TERM_CODONS = ("TTA", "TTG")   # docs/15: taken from Leu; COPY keeps CTT CTC CTA CTG


def dna2_table(table, uniform):
    """docs/15 fix 1: TERM gets TTA and TTG under the standard code and codon value 21 (CCC) under the uniform
    code, where 21 was the first NOP."""
    t = list(table)
    if uniform:
        t[21] = TERM
    else:
        for cod in DNA2_TERM_CODONS:
            v = codon_value(cod)
            assert t[v] == COPY
            t[v] = TERM
    return t


DNA3_HARVEST_CODONS = ("AGA", "AGG")   # docs/17 day 3: taken from Arg; L1 keeps CGT CGC CGA CGG


def dna3_table(table, uniform):
    """docs/17 day 3: HARVEST gets AGA and AGG under the standard code (the two Arg codons that vertebrate
    mitochondria reassign) and codon value 22 under the uniform code, the first NOP after TERM's 21."""
    t = list(table)
    if uniform:
        t[22] = HARVEST
    else:
        for cod in DNA3_HARVEST_CODONS:
            v = codon_value(cod)
            assert t[v] == L1
            t[v] = HARVEST
    return t


DNA3_SENSE_WAIT_CODONS = ("AGT", "AGC")   # docs/17 day 4: taken from Ser; R1 keeps TCT TCC TCA TCG


def dna3_day4_table(table, uniform):
    """docs/17 day 4: SENSE on AGT and WAIT on AGC (uniform code: codon values 23 and 24)."""
    t = list(table)
    if uniform:
        t[23], t[24] = SENSE, WAIT
    else:
        for cod, op in zip(DNA3_SENSE_WAIT_CODONS, (SENSE, WAIT)):
            v = codon_value(cod)
            assert t[v] == R1
            t[v] = op
    return t


DNA3_DAY5_CODONS = (("CCA", "CCG", "SETP", "EAT"), ("ACA", "ACG", "SETQ", "MOVE"), ("GTA", "GTG", "JNZB", "EXEC"))


def dna3_day5_table(table, uniform):
    """docs/17 day 5: EAT on CCA, CCG (Pro keeps CCT, CCC for SETP), MOVE on ACA, ACG (Thr keeps ACT, ACC for SETQ),
    EXEC on GTA, GTG (Val keeps GTT, GTC for JNZB); uniform code: codon values 25, 26, 27."""
    t = list(table)
    if uniform:
        t[25], t[26], t[27] = EAT, MOVE, EXEC
        return t
    for a, b, old, new in DNA3_DAY5_CODONS:
        for cod in (a, b):
            v = codon_value(cod)
            assert t[v] == OP_INDEX[old]
            t[v] = OP_INDEX[new]
    return t


DNA3_DAY6_CODONS = (("CGA", "L1", "SIGNAL"), ("CGG", "L1", "LISTEN"), ("TCA", "R1", "IN"), ("TCG", "R1", "OUT"))
# docs/17 day 6: the task ladder on a 2-bit input x: (name, function, bonus rank); bonus = task_bonus << rank
DNA3_TASKS = (("x^1", lambda x: x ^ 1, 0), ("x+1", lambda x: (x + 1) & 3, 0), ("x^3", lambda x: x ^ 3, 2),
              ("x-1", lambda x: (x + 3) & 3, 2))


def dna3_day6_table(table, uniform):
    """docs/17 day 6: SIGNAL on CGA and LISTEN on CGG (Arg; L1 keeps CGT, CGC), IN on TCA and OUT on TCG (Ser; R1 keeps
    TCT, TCC); uniform code: codon values 28, 29, 30, 31."""
    t = list(table)
    if uniform:
        t[28], t[29], t[30], t[31] = SIGNAL, LISTEN, IN, OUT
        return t
    for cod, old, new in DNA3_DAY6_CODONS:
        v = codon_value(cod)
        assert t[v] == OP_INDEX[old]
        t[v] = OP_INDEX[new]
    return t


def tri(u, P):
    """docs/17 day 4: integer triangle wave on [0, P): P - |4u - 2P|, in [-P, P]."""
    return P - abs(4 * u - 2 * P)


# docs/17 day 2: the eight neighbours of a slot on the grid, (dx, dy) with y growing downwards, in scan order
NEIGHBOURS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
SEA, LAND, FIRMAMENT = 0, 1, 2


def make_terrain(seed, S, W, firm, day3, land_cell, land_thr):
    """docs/17 days 2 and 3: terrain code per slot. Firmament rows are the rows y with y mod (R/2) < firm (two
    bands on the torus, so they cut it into two zones). On day 3 every other slot is land when the value noise
    of lattice spacing land_cell is at least land_thr / 65536 of its range, else sea; without day 3 it is sea."""
    R = S // W
    ter = bytearray(S)
    half = R // 2
    if day3:
        L = land_cell
        nx, ny = W // L, R // L
        lat = [rnd(seed, -2, i, 9) >> 48 for i in range(nx * ny)]      # 16-bit lattice values
        thr = land_thr * L * L
    for y in range(R):
        if firm and y % half < firm:
            for x in range(W):
                ter[y * W + x] = FIRMAMENT
            continue
        if not day3:
            continue
        j0, fy = divmod(y, L)
        j1 = (j0 + 1) % ny
        for x in range(W):
            i0, fx = divmod(x, L)
            i1 = (i0 + 1) % nx
            v = ((lat[j0 * nx + i0] * (L - fx) + lat[j0 * nx + i1] * fx) * (L - fy)
                 + (lat[j1 * nx + i0] * (L - fx) + lat[j1 * nx + i1] * fx) * fy)
            if v >= thr:
                ter[y * W + x] = LAND
    return ter


def splitmix64(x):
    """The splitmix64 finaliser of section 7 (x is taken modulo 2^64)."""
    z = (x + C1) & MASK64
    z = ((z ^ (z >> 30)) * C2) & MASK64
    z = ((z ^ (z >> 27)) * C3) & MASK64
    return z ^ (z >> 31)


def rnd(seed, t, k, j):
    """rnd(t, k, j) = splitmix64(seed xor mix(t, k, j)), t = -1 allowed."""
    mix = ((t + 1) * C1 + k * C2 + j * C3) & MASK64
    return splitmix64(seed ^ mix)


def fnv1a(data, h=FNV_OFFSET):
    """FNV-1a 64 over an iterable of bytes (ints 0..255), continuing from h."""
    p = FNV_PRIME
    m = MASK64
    for b in data:
        h = ((h ^ b) * p) & m
    return h


def pack2(cells):
    """Section 1 layout: cell i in byte i >> 2 at bits 2*(i & 3). cells: bytes-like of values 0..3."""
    n = len(cells)
    nbytes = (n + 3) >> 2
    if n & 3:
        cells = bytes(cells) + bytes(4 * nbytes - n)
    # the four interleaved lanes hold values < 4, so the shifted lanes never carry into the next byte
    a0 = int.from_bytes(cells[0::4], "little")
    a1 = int.from_bytes(cells[1::4], "little") << 2
    a2 = int.from_bytes(cells[2::4], "little") << 4
    a3 = int.from_bytes(cells[3::4], "little") << 6
    return (a0 | a1 | a2 | a3).to_bytes(nbytes, "little")


def gene_len(fwd, is_stop, base, maxc):
    """Codons of the slot at `base` from codon 0 up to (excluding) the first STOP; maxc if there is none."""
    seg = fwd[base:base + 3 * maxc]
    k = 0
    for a, b, c in zip(seg[0::3], seg[1::3], seg[2::3]):
        if is_stop[a * 16 + b * 4 + c]:
            return k
        k += 1
    return maxc


def is_expressible(fwd, table, is_stop, base, maxc):
    """Section 2: codon 0 is START (of the active table) and the gene has at least 2 codons."""
    if table[fwd[base] * 16 + fwd[base + 1] * 4 + fwd[base + 2]] != START:
        return False
    return gene_len(fwd, is_stop, base, maxc) >= 2


def gene_hash(fwd, is_stop, base, maxc):
    """Section 7: FNV-1a 64 over the codon values (one byte each) of the gene, START included, STOP excluded."""
    n = gene_len(fwd, is_stop, base, maxc)
    seg = fwd[base:base + 3 * n]
    h = FNV_OFFSET
    p = FNV_PRIME
    m = MASK64
    for a, b, c in zip(seg[0::3], seg[1::3], seg[2::3]):
        h = ((h ^ (a * 16 + b * 4 + c)) * p) & m
    return h


def snapshot_window(fwd, fwd_old, N, start, length, ranges):
    """The cells start .. start+length-1 (mod N) as one processor sees them: fwd_old everywhere except in the
    half-open cell ranges of `ranges` (its own slot, its claimed child slot), which are read live. The window may
    wrap around the ring, and tiles it if length > N. [13: SEEK]"""
    start %= N
    w = bytearray()
    pos, rem = start, length
    while rem > 0:
        take = min(rem, N - pos)
        w += fwd_old[pos:pos + take]
        pos = (pos + take) % N
        rem -= take
    end = start + length                      # unwrapped end
    for lo, hi in ranges:
        if lo >= hi:
            continue
        m_min = (start - hi) // N + 1         # smallest m with hi + m*N > start
        m_max = (end - 1 - lo) // N           # largest m with lo + m*N < end
        for m in range(m_min, m_max + 1):
            a = max(lo + m * N, start)
            b = min(hi + m * N, end)
            if a < b:
                w[a - start:b - start] = fwd[a - m * N:b - m * N]
    return w


def parse_genome(text):
    """'ATG GCT CCT' (or 'ATGGCTCCT') -> list of codon values."""
    toks = text.replace(",", " ").split()
    if len(toks) == 1 and len(toks[0]) > 3:
        s = toks[0]
        if len(s) % 3:
            raise ValueError("genome length is not a multiple of 3")
        toks = [s[i:i + 3] for i in range(0, len(s), 3)]
    return [codon_value(tok) for tok in toks]


def parse_auto(text):
    return None if str(text).lower() == "auto" else int(text)


class World:
    """The DNA-1 world of docs/09. Processors are stored densely by slot (lists indexed by slot)."""

    def __init__(self, N, G, I=16, T=64, mu=0.001, D=None, F=None, double=True, uniform=False, blank=False,
                 seed=1, genome=None, n_seed=0, dna2=False, days=(), seed_slots=None, seed_terrain="any",
                 W=0, firm=1, day_len=32, light_len=None, lmax=16, emax=1024, eshift=4, e_seed=512, e_spont=128,
                 g_pass=16, gp_sea=8, gp_land=0, gh_sea=16, gh_land=32, c_copy=1, maint=1, inflow_land=False,
                 land_cell=32, land_thr=32768, year_days=16, moon_days=8, season_amp=None, lat_amp=8, lmoon=1,
                 shade=False, c_move=2, c_fly=8, eat_shift=1, eat_eff=8, c_walk=4, task_bonus=8):
        if G < 6:
            raise ValueError("G must be at least 6 (two codons)")                      # [13: Limits]
        if N % G:
            raise ValueError("N must be a multiple of G")
        self.N, self.G = N, G
        self.S = S = N // G
        if S < 2:
            raise ValueError("at least 2 slots are needed (ALLOC draws modulo S - 1)")  # [13: Limits]
        self.maxc = G // 3                      # usable codons per slot; last index maxc - 1
        self.I, self.T, self.mu = I, T, float(mu)
        self.D = max(1, N // (1 << 20)) if D is None else D
        self.F = max(1, S // (1 << 12)) if F is None else F
        self.double = double
        self.seed = seed & MASK64
        self.table = uniform_table() if uniform else standard_table()
        # docs/17: DNA-3 days; every day implies DNA-2, day 3 needs days 1 and 2
        days = tuple(sorted(set(days)))
        if any(d not in (1, 2, 3, 4, 5, 6) for d in days):
            raise ValueError("only days 1 to 6 are implemented")
        if 6 in days and not all(d in days for d in (1, 2, 3, 4, 5)):
            raise ValueError("day 6 needs days 1 to 5")
        if 5 in days and not all(d in days for d in (1, 2, 3, 4)):
            raise ValueError("day 5 needs days 1 to 4")
        if 3 in days and not (1 in days and 2 in days):
            raise ValueError("day 3 needs days 1 and 2")
        if 4 in days and not (1 in days and 2 in days and 3 in days):
            raise ValueError("day 4 needs days 1, 2 and 3")
        self.days = days
        self.d1, self.d2, self.d3, self.d4, self.d5 = 1 in days, 2 in days, 3 in days, 4 in days, 5 in days
        self.d6 = 6 in days
        dna2 = dna2 or bool(days)
        self.dna2 = dna2
        if dna2:
            self.table = dna2_table(self.table, uniform)
        if self.d3:
            self.table = dna3_table(self.table, uniform)
        if self.d4:
            self.table = dna3_day4_table(self.table, uniform)
        if self.d5:
            self.table = dna3_day5_table(self.table, uniform)
        if self.d6:
            self.table = dna3_day6_table(self.table, uniform)
        if c_walk < 0 or task_bonus < 0:
            raise ValueError("bad day-6 parameters")
        self.c_walk, self.task_bonus = c_walk, task_bonus
        if min(c_move, c_fly, eat_shift, eat_eff) < 0 or eat_eff > 16:
            raise ValueError("bad day-5 parameters")
        self.c_move, self.c_fly, self.eat_shift, self.eat_eff = c_move, c_fly, eat_shift, eat_eff
        light_len = day_len // 2 if light_len is None else light_len
        if self.d1:
            if day_len < 1 or not 0 <= light_len <= day_len:
                raise ValueError("need day_len >= 1 and 0 <= light_len <= day_len")
            if T < 0 or eshift < 0 or emax < 1 or not (0 <= e_seed <= emax) or not (0 <= e_spont <= emax) \
                    or min(lmax, g_pass, gp_sea, gp_land, gh_sea, gh_land, c_copy, maint) < 0:
                raise ValueError("bad energy parameters")
        elif T < 1:
            raise ValueError("T must be at least 1 (T = 0, no age limit, exists only on day 1)")
        self.day_len, self.light_len = day_len, light_len
        # docs/17 day 4: calendar
        season_amp = day_len // 4 if season_amp is None else season_amp
        if self.d4 and (year_days < 1 or moon_days < 1 or season_amp < 0 or season_amp > light_len
                        or light_len + season_amp > day_len or not 0 <= lat_amp <= 16 or lmoon < 0):
            raise ValueError("bad day-4 parameters (need 0 <= season_amp <= light_len, light_len + season_amp <= day_len, "
                             "0 <= lat_amp <= 16)")
        self.year_days, self.moon_days, self.season_amp, self.lat_amp, self.lmoon = \
            year_days, moon_days, season_amp, lat_amp, lmoon
        self.lmax, self.emax, self.eshift, self.e_seed, self.e_spont = lmax, emax, eshift, e_seed, e_spont
        self.g_pass, self.c_copy, self.maint, self.inflow_land = g_pass, c_copy, maint, inflow_land
        if shade and not self.d3:
            raise ValueError("--shade is a day-3 option")
        self.shade = shade                      # docs/17 section 14: light shared with the live neighbours
        self.gp = (gp_sea, gp_land) if self.d3 else (g_pass, g_pass)     # passive gain by terrain, sixteenths
        self.gh = (gh_sea, gh_land)                                      # HARVEST gain by terrain, sixteenths
        if self.d2:
            if W < 3 or S % W or S // W < 3:
                raise ValueError("day 2 needs --W >= 3 dividing S with S / W >= 3 rows")
            R = S // W
            if firm < 0 or (firm and (R % 2 or firm > R // 2 - 1)):
                raise ValueError("the firmament needs an even number of rows R and 0 < firm <= R/2 - 1")
            if self.d3 and (land_cell < 1 or W % land_cell or R % land_cell):
                raise ValueError("--land-cell must divide W and R")
            self.W, self.R, self.firm = W, R, firm
            self.terrain = make_terrain(self.seed, S, W, firm, self.d3, land_cell, land_thr)
        else:
            self.W, self.R, self.firm = 0, 0, 0
            self.terrain = None
        self.is_stop = bytes(1 if op == STOP else 0 for op in self.table)
        # execution table: START inside a gene is NOP; a fetched STOP (only possible when the gene has
        # degenerated to len <= 1 during the tick) is NOP too [13: Degenerate genes]
        self.xtable = [NOP if op in (START, STOP) else op for op in self.table]
        self.stop_codon = min(c for c in range(64) if self.table[c] == STOP)             # [13: Seeding]

        # section 6: the tape
        fwd = bytearray(N)
        if not blank:
            # fwd[i] = rnd(-1, i, 0) mod 4; mix(-1, i, 0) = i * C2, built incrementally
            seed_ = self.seed
            m = MASK64
            x = 0
            for i in range(N):
                z = ((seed_ ^ x) + C1) & m
                z = ((z ^ (z >> 30)) * C2) & m
                z = ((z ^ (z >> 27)) * C3) & m
                fwd[i] = (z ^ (z >> 31)) & 3
                x = (x + C2) & m
        self.fwd = fwd
        self.rev = fwd.translate(COMP_TABLE)

        # processors, dense by slot
        self.alive = bytearray(S)
        self.pc = [0] * S
        self.H = [0] * S
        self.P = [0] * S
        self.Q = [0] * S
        self.child = [-1] * S
        self.A = [0] * S
        self.age = [0] * S
        self.depth = [0] * S
        self.copied = [0] * S
        self.E = [0] * S                        # docs/17 day 1: energy store
        self.div_energy = [0] * S               # endowment of the child marked by DIVIDE this tick
        # docs/17 energy ledger (cumulative), in this order: passive light, HARVEST, spontaneous births' energy, spent
        # on instructions, on copied cells, on maintenance, lost with the dead, lost endowments (child not born),
        # light wasted on full stores (not part of the balance)
        self.ledger = [0] * 13                  # day 5 adds: eaten (moved, not lost), trophic loss, movement; day 6: tasks
        self.sig = [0] * S                      # docs/17 day 6: signal register
        self.xin = [0] * S                      # docs/17 day 6: the input of the last IN (0..3)
        self.tmask = [0] * S                    # docs/17 day 6: 1 while the last input awaits its one answer
        self.streak = [0] * S                   # docs/17 day 6: per task k a 2-bit count of correct answers in a row
        self.task_count = [0] * len(DNA3_TASKS)
        self.xs = [0] * S                       # docs/17 day 5: the slot whose gene the processor executes
        self.energy_init = 0
        self.starved = 0
        self.claimed_by = [-1] * S              # parent slot that holds the claim on this slot, or -1
        self.live = set()
        self.births = self.faithful = self.spontaneous = self.deaths = 0
        self.claims = self.copied_cells = 0     # diagnostics for the done line: ALLOC grants, cells written by COPY/COPYR
        self.birth_log = []                     # (tick, parent, child, parent_hash or None, child_hash, faithful)
        self.tick_no = 0

        # seeded organisms
        if n_seed > 0:
            if not genome:
                raise ValueError("--n-seed needs --genome")
            if len(genome) > self.maxc:
                raise ValueError("genome of %d codons does not fit a slot of %d codons" % (len(genome), self.maxc))
            if n_seed > S:
                raise ValueError("more seeds than slots")
            step = S // n_seed                                                          # [13: Seeding]
            if seed_slots is not None:
                if len(seed_slots) != n_seed or len(set(seed_slots)) != n_seed or any(not 0 <= x < S for x in seed_slots):
                    raise ValueError("--seed-slots needs n_seed distinct slots in [0, S)")
            chosen = []
            used = set()
            for k in range(n_seed):
                if seed_slots is not None:
                    s = seed_slots[k]
                    if self.terrain is not None and self.terrain[s] == FIRMAMENT:
                        raise ValueError("seed slot %d lies in the firmament" % s)
                else:
                    s = k * step
                    if self.terrain is not None:
                        # docs/17: advance to the next slot that is not firmament, matches --seed-terrain and is unseeded
                        want = {"any": (SEA, LAND), "sea": (SEA,), "land": (LAND,)}[seed_terrain]
                        n_try = 0
                        while self.terrain[s] not in want or s in used:
                            s = (s + 1) % S
                            n_try += 1
                            if n_try > S:
                                raise ValueError("no slot for seed %d" % k)
                used.add(s)
                chosen.append(s)
            for s in chosen:
                base = s * G
                for ci in range(self.maxc):
                    c = genome[ci] if ci < len(genome) else self.stop_codon
                    for off, v in ((0, c >> 4), (1, (c >> 2) & 3), (2, c & 3)):
                        fwd[base + 3 * ci + off] = v
                        self.rev[base + 3 * ci + off] = v ^ 3
                # the G mod 3 trailing cells of the slot are not codons and keep their soup content [13: Seeding]
                self.born(s, 0)                 # unconditional, not logged, not counted [13: Seeding, Counters]
                if self.d1:
                    self.E[s] = e_seed
                    self.energy_init += e_seed

    # -- helpers ---------------------------------------------------------------------------------------------
    def born(self, s, depth):
        """A new processor in slot s with the initial values of section 4."""
        self.xs[s] = s
        self.sig[s] = self.xin[s] = self.tmask[s] = self.streak[s] = 0
        self.alive[s] = 1
        self.live.add(s)
        self.pc[s] = 1
        self.H[s] = self.P[s] = s * self.G
        self.Q[s] = 0
        self.child[s] = -1
        self.A[s] = 0
        self.age[s] = 0
        self.depth[s] = depth
        self.copied[s] = 0

    def expressible(self, s):
        return is_expressible(self.fwd, self.table, self.is_stop, s * self.G, self.maxc)

    def genehash(self, s):
        return gene_hash(self.fwd, self.is_stop, s * self.G, self.maxc)

    def state_bytes(self):
        """Section 8: packed fwd, packed rev, per live processor (slot order) ten LE u64 fields, four counters.
        docs/17: day 1 appends the energy E as an eleventh field, day 3 the starvation count as a fifth counter."""
        out = bytearray(pack2(self.fwd))
        out += pack2(self.rev)
        pk = struct.Struct("<10Q").pack
        pe = struct.Struct("<Q").pack
        d1 = self.d1
        for s in sorted(self.live):
            out += pk(s, self.pc[s], self.H[s], self.P[s], self.Q[s], self.child[s] & MASK64,   # -1 -> 2^64 - 1
                      self.A[s], self.age[s], self.depth[s], self.copied[s])
            if d1:
                out += pe(self.E[s])
            if self.d5:
                out += pe(self.xs[s])           # docs/17 day 5: the executed slot, a twelfth field
            if self.d6:                         # docs/17 day 6: signal, input, open tasks
                out += pe(self.sig[s]) + pe(self.xin[s]) + pe(self.tmask[s]) + pe(self.streak[s])
        out += struct.pack("<4Q", self.births, self.faithful, self.spontaneous, self.deaths)
        if self.d3:
            out += pe(self.starved)
        return bytes(out)

    def light(self, t):
        """docs/17 day 1: the light level of tick t (square wave: lmax for the first light_len ticks of each day)."""
        return self.lmax if t % self.day_len < self.light_len else 0

    def sky(self, t, s):
        """docs/17 day 4: (light, [sun, moon, season, stars]) at tick t in slot s."""
        Py, Pm, R = self.year_days, self.moon_days, self.R
        d, phi = divmod(t, self.day_len)
        y = s // self.W
        sy = tri(d % Py, Py) + Py
        sz = 2 * Py - sy if y < R // 2 else sy
        Lz = self.light_len - self.season_amp + (self.season_amp * sz) // Py
        up = phi < Lz
        ys = R // 4 + ((R // 2) * sy) // (2 * Py)
        dy = abs(y - ys)
        dist = min(dy, R - dy)
        fac = 16 - (self.lat_amp * dist) // (R // 2)
        ml = ((tri(d % Pm, Pm) + Pm) * 3 + Pm) // (2 * Pm)
        lam = (self.lmax * fac) // 16 if up else self.lmoon * ml
        sun = 0 if not up else (3 if fac >= 14 else 2 if fac >= 11 else 1)
        season = (4 * sz) // (2 * Py + 1)
        stars = 0 if up else 1 + (3 * y) // R
        return lam, (sun, ml, season, stars)

    def has_harvest(self, s):
        return self.has_op(s, HARVEST)

    def has_op(self, s, opcode):
        """docs/17 days 3 and 4: the gene of slot s (codons 1 .. len-1) contains a codon of this operation."""
        base = s * self.G
        n = gene_len(self.fwd, self.is_stop, base, self.maxc)
        f, tb = self.fwd, self.table
        for k in range(1, n):
            i = base + 3 * k
            if tb[f[i] * 16 + f[i + 1] * 4 + f[i + 2]] == opcode:
                return True
        return False

    def state_hash(self):
        return fnv1a(self.state_bytes())

    def census(self, t, since, state_hash=None):
        """Section 5 step 8 for the state after tick t (the "tick" field is t). `since` is the (births, faithful,
        spontaneous, deaths) at the previous census. [13: Census schedule]"""
        hashes = {}
        dhist = {}
        for s in self.live:
            h = self.genehash(s)
            hashes[h] = hashes.get(h, 0) + 1
            d = self.depth[s]
            dhist[d] = dhist.get(d, 0) + 1
        top = sorted(hashes.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        out = {
            "tick": t,
            "live": len(self.live),
            "births": self.births - since[0],
            "faithful": self.faithful - since[1],
            "spontaneous": self.spontaneous - since[2],
            "deaths": self.deaths - since[3],
            "births_total": self.births,
            "faithful_total": self.faithful,
            "spontaneous_total": self.spontaneous,
            "deaths_total": self.deaths,
            "distinct_genes": len(hashes),
            "top_genes": [["%016x" % h, c] for h, c in top],
            "depth_hist": [[d, c] for d, c in sorted(dhist.items())],
        }
        if self.d1:                                     # docs/17 census additions, in this key order
            # day 4: the light at column 0 of the middle row of each zone, a list of two
            out["light"] = self.light(t) if not self.d4 else [self.sky(t, (self.R // 4) * self.W)[0],
                                                              self.sky(t, (self.R // 2 + self.R // 4) * self.W)[0]]
            out["energy_total"] = sum(self.E[s] for s in self.live)
            out["energy_init"] = self.energy_init
            out["ledger"] = list(self.ledger[:13 if self.d6 else 12 if self.d5 else 9])
        if self.d2 and self.firm:
            z = [0, 0]
            for s in self.live:
                z[0 if s // self.W < self.R // 2 else 1] += 1
            out["zone_live"] = z
        if self.d3:
            out["starved_total"] = self.starved
            out["live_land"] = sum(1 for s in self.live if self.terrain[s] == LAND)
            out["harvesters"] = sum(1 for s in self.live if self.has_harvest(s))
        if self.d4:                                     # docs/17 day 4
            out["sensers"] = sum(1 for s in self.live if self.has_op(s, SENSE))
            out["sleepers"] = sum(1 for s in self.live if self.has_op(s, WAIT))
        if self.d5:                                     # docs/17 day 5
            out["eaters"] = sum(1 for s in self.live if self.has_op(s, EAT))
            out["movers"] = sum(1 for s in self.live if self.has_op(s, MOVE))
            out["execers"] = sum(1 for s in self.live if self.has_op(s, EXEC))
            out["foreign"] = sum(1 for s in self.live if self.xs[s] != s)
        if self.d6:                                     # docs/17 day 6
            out["signalers"] = sum(1 for s in self.live if self.has_op(s, SIGNAL))
            out["listeners"] = sum(1 for s in self.live if self.has_op(s, LISTEN))
            out["taskers"] = sum(1 for s in self.live if self.has_op(s, OUT))
            out["tasks"] = list(self.task_count)
        out["hash"] = None if state_hash is None else "%016x" % state_hash
        return out

    # -- the tick ---------------------------------------------------------------------------------------------
    def tick(self):
        """One tick, the eight steps of section 5 in order."""
        t = self.tick_no
        N, G, S, I, maxc = self.N, self.G, self.S, self.I, self.maxc
        fwd, rev = self.fwd, self.rev
        xtable, is_stop = self.xtable, self.is_stop
        pc_, H_, P_, Q_, child_, A_, copied_ = self.pc, self.H, self.P, self.Q, self.child, self.A, self.copied
        alive, claimed_by, live = self.alive, self.claimed_by, self.live
        seed, mu = self.seed, self.mu
        dna2 = self.dna2
        d1, d2, d3 = self.d1, self.d2, self.d3
        E_ = self.E
        ledger = self.ledger
        lam = self.light(t) if d1 else 0
        d4 = self.d4
        emax, eshift, c_copy = self.emax, self.eshift, self.c_copy
        terrain = self.terrain
        win = 3 * G + 2                         # SEEK window: candidates H+1 .. H+3G, each needs 3 cells
        alloc_req = []                          # requesting slots, increasing
        marked = []                             # (parent, child) pairs marked by DIVIDE, increasing parent
        d5 = self.d5
        eat_req = {}                            # docs/17 day 5: eater slot -> prey slot
        d6 = self.d6
        sig_old = list(self.sig) if d6 else None   # docs/17 day 6: signals as they were at the start of the tick
        move_req = {}                           # docs/17 day 5: mover slot -> operand codon
        xs_ = self.xs

        # 1. execute
        if live:
            fwd_old = bytes(fwd)                # the snapshot of section 4
            for s in sorted(live):
                base = s * G
                own_hi = base + G
                pc = pc_[s]; H = H_[s]; P = P_[s]; Q = Q_[s]; child = child_[s]; A = A_[s]; copied = copied_[s]
                if child >= 0:
                    ch_lo = child * G
                    ch_hi = ch_lo + G
                else:
                    ch_lo = ch_hi = -1          # an empty range: no cell satisfies ch_lo <= x < ch_hi
                L = gene_len(fwd, is_stop, base, maxc)
                # docs/17 day 5: the code being executed: own gene (live) or a neighbour's (tick-start snapshot)
                xs = xs_[s] if d5 else s
                csrc, cb0 = fwd, base
                if xs != s:
                    if is_expressible(fwd_old, self.table, is_stop, xs * G, maxc):
                        csrc, cb0 = fwd_old, xs * G
                        L = gene_len(fwd_old, is_stop, cb0, maxc)
                        if pc >= L:             # the foreign gene got shorter: home
                            xs, csrc, cb0, pc = s, fwd, base, 1
                            L = gene_len(fwd, is_stop, base, maxc)
                    else:                       # the foreign gene is gone: home
                        xs, pc = s, 1
                ate = False
                requested = False
                budget = I
                if d1:
                    # docs/17 day 1: passive light first, then the budget min(I, E >> eshift); one unit per instruction
                    E = E_[s]
                    ter = terrain[s] if terrain is not None else SEA
                    if d4:
                        lam, chan = self.sky(t, s)      # docs/17 day 4: light and sky channels of this slot
                    g = (lam * self.gp[ter]) >> 4
                    if self.shade:
                        # docs/17 section 14: n = live neighbours at the start of the tick; every light gain of this
                        # processor in this tick is multiplied by (9 - n) / 9 (floor)
                        Wd, Rd = self.W, self.R
                        x0, y0 = s % Wd, s // Wd
                        nl = 0
                        for dx, dy in NEIGHBOURS:
                            nl += alive[((y0 + dy) % Rd) * Wd + (x0 + dx) % Wd]
                        shade_k = 9 - nl
                        g = (g * shade_k) // 9
                    add = min(g, emax - E)
                    E += add
                    ledger[0] += add
                    ledger[8] += g - add
                    budget = min(I, E >> eshift)
                    harvested = False
                for _ in range(budget):
                    if d1:
                        if E < 1:
                            break
                        E -= 1
                        ledger[3] += 1
                    cb = cb0 + 3 * pc
                    op = xtable[csrc[cb] * 16 + csrc[cb + 1] * 4 + csrc[cb + 2]]
                    pc += 1
                    if op == COPY or op == COPYR:
                        if dna2 and child < 0:
                            pass                # docs/15 fix 3: no claimed child, nothing happens (P and Q stay)
                        else:
                          v = fwd[P] if (base <= P < own_hi or ch_lo <= P < ch_hi) else fwd_old[P]
                          if op == COPYR:
                              v ^= 3
                          if ch_lo <= Q < ch_hi:  # child claimed and Q inside it: write
                              if d3:
                                  if E < c_copy:  # docs/17 day 3: no energy for the cell: back to this COPY, tick over
                                      pc -= 1
                                      break
                                  E -= c_copy
                                  ledger[4] += c_copy
                              if mu > 0.0:
                                  r = rnd(seed, t, s, 1000 + copied)                      # [13: Mutation index]
                                  if (r >> 32) / 4294967296.0 < mu:                       # [13: Mutation test]
                                      v = (v + 1 + r % 3) & 3
                              fwd[Q] = v
                              rev[Q] = v ^ 3
                              copied += 1
                              self.copied_cells += 1
                          P = (P + 1) % N if op == COPY else (P - 1) % N
                          Q = (Q + 1) % N
                    elif op == R1:
                        H = (H + 1) % N
                    elif op == L1:
                        H = (H - 1) % N
                    elif op == JNZB:
                        v = fwd[H] if (base <= H < own_hi or ch_lo <= H < ch_hi) else fwd_old[H]
                        if v:
                            pc -= 2
                            if pc < 1:
                                pc = 1
                    elif op == SETP:
                        P = H
                    elif op == SETQ:
                        Q = H
                    elif op == ALLOC:
                        if child < 0 and not requested:                                 # [13: Claim release]
                            requested = True
                            alloc_req.append(s)
                    elif op == DIVIDE:
                        if child >= 0 and copied >= 6:
                            marked.append((s, child))   # the slot stays claimed until step 3 [13: Claim release]
                            child = -1
                            copied = 0
                            Q = 0
                            ch_lo = ch_hi = -1
                            if dna2:
                                P = base        # docs/15 fix 2: the template is re-armed for the next copy
                            if d1:              # docs/17 day 1: half of the store goes with the child
                                x = E >> 1
                                E -= x
                                self.div_energy[s] = x
                    elif op == SKNZ:
                        v = fwd[H] if (base <= H < own_hi or ch_lo <= H < ch_hi) else fwd_old[H]
                        if v:
                            pc += 1
                    elif op == SKZ:
                        v = fwd[H] if (base <= H < own_hi or ch_lo <= H < ch_hi) else fwd_old[H]
                        if not v:
                            pc += 1
                    elif op == LDA:
                        A = fwd[H] if (base <= H < own_hi or ch_lo <= H < ch_hi) else fwd_old[H]
                    elif op == STA or op == INC or op == FLIP:
                        if base <= H < own_hi:
                            v = A if op == STA else ((fwd[H] + 1) & 3 if op == INC else fwd[H] ^ 1)
                            fwd[H] = v
                            rev[H] = v ^ 3
                            if xs == s:         # docs/17 day 5: only the code being executed sets L
                                L = gene_len(fwd, is_stop, base, maxc)                  # [13: Degenerate genes]
                        elif ch_lo <= H < ch_hi:
                            v = A if op == STA else ((fwd[H] + 1) & 3 if op == INC else fwd[H] ^ 1)
                            fwd[H] = v
                            rev[H] = v ^ 3
                        # else: NOP (write rule)
                    elif op == R3:
                        H = (H + 3) % N
                    elif op == L3:
                        H = (H - 3) % N
                    elif op == SEEK:
                        if pc < L:              # the template codon lies inside the gene
                            tb = cb0 + 3 * pc
                            pat = bytes((csrc[tb] ^ 3, csrc[tb + 1] ^ 3, csrc[tb + 2] ^ 3))
                            pc += 1             # the template is consumed
                            w = snapshot_window(fwd, fwd_old, N, H + 1, win, ((base, own_hi), (ch_lo, ch_hi)))
                            d = w.find(pat)
                            if d >= 0:          # d <= 3G - 1 by construction of the window length
                                H = (H + 1 + d) % N
                        # else: NOP
                    elif op == REWIND:
                        P = base
                        if child >= 0:
                            Q = child * G
                    elif op == TERM:
                        # docs/15 fix 1: skip the next codon unless the template has just passed a STOP codon,
                        # read codon-aligned in P's slot: o = P mod G, o >= 3, o % 3 == 0, o / 3 <= maxc, and the
                        # codon of cells P-3 .. P-1 is STOP; cells read under the snapshot rule
                        o = P % G
                        term = False
                        if o >= 3 and o % 3 == 0 and o // 3 <= maxc:
                            c3 = 0
                            for i in (P - 3, P - 2, P - 1):
                                c3 = c3 * 4 + (fwd[i] if (base <= i < own_hi or ch_lo <= i < ch_hi) else fwd_old[i])
                            term = bool(is_stop[c3])
                        if not term:
                            pc += 1
                    elif op == SENSE:
                        # docs/17 day 4: the next codon is the operand (consumed); skip the codon after it if the
                        # sky channel (operand mod 4) is nonzero; NOP when the operand lies beyond the gene end
                        if pc < L:
                            ob = cb0 + 3 * pc
                            c = csrc[ob] * 16 + csrc[ob + 1] * 4 + csrc[ob + 2]
                            pc += 1
                            if chan[c & 3]:
                                pc += 1
                    elif op == WAIT:
                        # docs/17 day 4: the tick ends after this instruction (the wrap rule still applies)
                        if pc >= L:
                            if xs != s:
                                xs, csrc, cb0 = s, fwd, base
                                L = gene_len(fwd, is_stop, base, maxc)
                            pc = 1
                        break
                    elif op == SIGNAL:
                        self.sig[s] = A         # docs/17 day 6: visible to the neighbours from the next tick
                    elif op == IN:
                        # docs/17 day 6: a fresh input x, written into the I/O cell (the last cell of the own slot, both
                        # strands), the head moved there and A = x; it awaits one answer
                        A = rnd(seed, t, s, 4000) & 3
                        H = own_hi - 1
                        fwd[H] = A
                        rev[H] = A ^ 3
                        if xs == s:
                            L = gene_len(fwd, is_stop, base, maxc)
                        self.xin[s] = A
                        self.tmask[s] = 1
                    elif op == OUT:
                        # docs/17 day 6: the one answer to the last input; per task a correct answer raises its streak
                        # (to at most 3) and a wrong one resets it; a task pays its bonus when its streak is 3
                        if self.tmask[s]:
                            self.tmask[s] = 0
                            xi, st = self.xin[s], self.streak[s]
                            for k, (_n, f, rank) in enumerate(DNA3_TASKS):
                                c = (st >> (2 * k)) & 3
                                c = min(3, c + 1) if f(xi) == A else 0
                                st = (st & ~(3 << (2 * k))) | (c << (2 * k))
                                if c == 3:
                                    g = self.task_bonus << rank
                                    add = min(g, emax - E)
                                    E += add
                                    ledger[12] += add
                                    ledger[8] += g - add
                                    self.task_count[k] += 1
                            self.streak[s] = st
                    elif op == EAT or op == MOVE or op == EXEC or op == LISTEN:
                        # docs/17 day 5: the next codon is the operand (consumed); NOP if it lies beyond the code's end
                        if pc < L:
                            ob = cb0 + 3 * pc
                            c = csrc[ob] * 16 + csrc[ob + 1] * 4 + csrc[ob + 2]
                            pc += 1
                            dx, dy = NEIGHBOURS[c & 7]
                            if op == LISTEN:
                                # docs/17 day 6: A = the neighbour's signal at the start of the tick (0 if no processor)
                                nb = ((s // self.W + dy) % self.R) * self.W + (s % self.W + dx) % self.W
                                A = sig_old[nb] if alive[nb] else 0
                            elif op == EAT:
                                # one EAT per tick, aimed at a neighbour that is alive (live flags of the tick start)
                                nb = ((s // self.W + dy) % self.R) * self.W + (s % self.W + dx) % self.W
                                if not ate and alive[nb]:
                                    ate = True
                                    eat_req[s] = nb
                            elif op == EXEC:
                                # run the neighbour's gene from its codon 1 (read from the tick-start snapshot)
                                nb = ((s // self.W + dy) % self.R) * self.W + (s % self.W + dx) % self.W
                                if is_expressible(fwd_old, self.table, is_stop, nb * G, maxc):
                                    xs, csrc, cb0, pc = nb, fwd_old, nb * G, 1
                                    L = gene_len(fwd_old, is_stop, cb0, maxc)
                            elif child < 0:
                                # MOVE: a request resolved in step 2; it withdraws this tick's ALLOC and ends the tick
                                move_req[s] = c
                                if requested:
                                    alloc_req.remove(s)
                                    requested = False
                                if pc >= L:
                                    if xs != s:
                                        xs, csrc, cb0 = s, fwd, base
                                        L = gene_len(fwd, is_stop, base, maxc)
                                    pc = 1
                                break
                    elif op == HARVEST:
                        # docs/17 day 3: once per tick, light times the terrain's HARVEST gain
                        if not harvested:
                            harvested = True
                            g = (lam * self.gh[ter]) >> 4
                            if self.shade:
                                g = (g * shade_k) // 9
                            add = min(g, emax - E)
                            E += add
                            ledger[1] += add
                            ledger[8] += g - add
                    # NOP: nothing
                    if pc >= L:
                        if xs != s:             # docs/17 day 5: the foreign gene wrapped: home, own codon 1
                            xs, csrc, cb0 = s, fwd, base
                            L = gene_len(fwd, is_stop, base, maxc)
                        pc = 1
                pc_[s] = pc; H_[s] = H; P_[s] = P; Q_[s] = Q; child_[s] = child; A_[s] = A; copied_[s] = copied
                if d1:
                    E_[s] = E
                if d5:
                    xs_[s] = xs

        # 2a. docs/17 day 5: predation, on the stores after step 1; every prey's eaters in increasing key order
        if eat_req:
            E0 = {x: E_[x] for x in live}
            loss, gain = {}, {}
            by_prey = {}
            for e, pr in eat_req.items():
                by_prey.setdefault(pr, []).append((((rnd(seed, t, e, 3000) >> 32) << 32) | e, e))
            for pr in sorted(by_prey):
                rem = E0[pr]
                for _key, e in sorted(by_prey[pr]):
                    x = rem >> self.eat_shift
                    rem -= x
                    gn = (x * self.eat_eff) >> 4
                    gain[e] = gn
                    ledger[9] += x
                    ledger[10] += x - gn
                loss[pr] = E0[pr] - rem
            for x in sorted(set(loss) | set(gain)):
                v = E0[x] - loss.get(x, 0) + gain.get(x, 0)
                if v > emax:
                    ledger[10] += v - emax
                    v = emax
                E_[x] = v

        # 2. resolve ALLOC: increasing slot order, so the lowest requester takes a contested candidate
        Sm1 = S - 1
        if d2:
            # docs/17 day 2: every requester scans its eight neighbours from a random direction on the state after
            # step 1 and proposes the first free one (alive no, claimed no, not firmament); a contested slot goes to
            # the requester with the lowest key (rnd(t, s, 2000) >> 32) * 2^32 + s
            W, R = self.W, self.R
            best = {}
            move_cost = {}
            for s in sorted(move_req):
                # docs/17 day 5: a move of 1 to 4 slots in direction c mod 8; one slot = swimming (sea to sea, c_move),
                # more = flying (any target but the firmament, c_fly per slot); the target must be free and the store
                # must hold the cost
                c = move_req[s]
                dist = 1 + ((c >> 3) & 3)
                dx, dy = NEIGHBOURS[c & 7]
                tg = ((s // W + dy * dist) % R) * W + (s % W + dx * dist) % W
                cost = self.c_move if dist == 1 else self.c_fly * dist
                if dist == 1 and self.d6 and (terrain[s] == LAND or terrain[tg] == LAND):
                    cost = self.c_walk          # docs/17 day 6: walking on land (and stepping ashore or into the sea)
                if terrain[tg] == FIRMAMENT or alive[tg] or claimed_by[tg] >= 0 or E_[s] < cost:
                    continue
                if dist == 1 and not self.d6 and (terrain[s] != SEA or terrain[tg] != SEA):
                    continue
                key = ((rnd(seed, t, s, 2000) >> 32) << 32) | s
                if tg not in best or key < best[tg][0]:
                    best[tg] = (key, s)
                move_cost[s] = cost
            for s in alloc_req:
                x, y = s % W, s // W
                r = rnd(seed, t, s, 0) & 7
                for i in range(8):
                    dx, dy = NEIGHBOURS[(r + i) & 7]
                    c = ((y + dy) % R) * W + (x + dx) % W
                    if terrain[c] != FIRMAMENT and not alive[c] and claimed_by[c] < 0:
                        key = ((rnd(seed, t, s, 2000) >> 32) << 32) | s
                        if c not in best or key < best[c][0]:
                            best[c] = (key, s)
                        break
            moves = []
            for c in sorted(best):
                s = best[c][1]
                if s in move_cost and s in move_req:
                    moves.append((s, c))
                    continue
                claimed_by[c] = s
                child_[s] = c
                Q_[s] = c * G
                copied_[s] = 0
                self.claims += 1
            # 2c. docs/17 day 5: granted moves, in increasing source slot: the G cells and the processor go to the
            # target; H and P move with it; the source keeps its cells (a shell) and is dead
            for s, c in sorted(moves):
                sb, tb = s * G, c * G
                fwd[tb:tb + G] = fwd[sb:sb + G]
                rev[tb:tb + G] = rev[sb:sb + G]
                sh = (c - s) * G
                self.pc[c] = pc_[s]
                self.H[c] = (H_[s] + sh) % N
                self.P[c] = (P_[s] + sh) % N
                self.Q[c] = 0
                child_[c] = -1
                self.A[c] = self.A[s]
                self.sig[c], self.xin[c], self.tmask[c], self.streak[c] = self.sig[s], self.xin[s], self.tmask[s], self.streak[s]
                self.age[c] = self.age[s]
                self.depth[c] = self.depth[s]
                copied_[c] = 0
                E_[c] = E_[s] - move_cost[s]
                ledger[11] += move_cost[s]
                xs_[c] = c if xs_[s] == s else xs_[s]
                alive[s] = 0
                live.discard(s)
                alive[c] = 1
                live.add(c)
        else:
          for s in alloc_req:
            cand = (s + 1 + rnd(seed, t, s, 0) % Sm1) % S
            if not alive[cand] and claimed_by[cand] < 0:
                claimed_by[cand] = s
                child_[s] = cand
                Q_[s] = cand * G
                copied_[s] = 0
                self.claims += 1

        # 3. births
        for p, ch in marked:
            claimed_by[ch] = -1                 # the claim is released whether or not something is born
            if self.expressible(ch):
                self.born(ch, self.depth[p] + 1)
                if d1:
                    E_[ch] = self.div_energy[p]
                ph = self.genehash(p)           # the parent's gene as it is now, after its own writes of the tick
                chh = self.genehash(ch)
                f = ph == chh
                self.births += 1                                                        # [13: Counters]
                self.faithful += f
                self.birth_log.append((t, p, ch, ph, chh, f))
            elif d1:
                ledger[7] += self.div_energy[p]  # docs/17: the endowment of a child that is not born is lost

        # 4. deaths (age += 1 for every processor alive now, this tick's newborns included) [13: Age]
        T = self.T
        age_ = self.age
        maint = self.maint
        for s in sorted(live):
            age_[s] += 1
            dies = (age_[s] >= T and (T > 0 or not d1)) or not self.expressible(s)   # docs/17: T = 0 on day 1, no limit
            if not dies and d3:
                # docs/17 day 3: maintenance; a processor that cannot pay it starves
                if E_[s] < maint:
                    dies = True
                    self.starved += 1
                else:
                    E_[s] -= maint
                    ledger[5] += maint
            if dies:
                if d1:
                    ledger[6] += E_[s]
                alive[s] = 0
                live.discard(s)
                if child_[s] >= 0:
                    claimed_by[child_[s]] = -1
                    child_[s] = -1
                self.deaths += 1

        # 5. damage (rev is not touched)
        hit = []
        for k in range(self.D):
            i = rnd(seed, t, k, 1) % N
            fwd[i] ^= 1 + rnd(seed, t, k, 2) % 3
            hit.append(i)

        # 6. repair from the reverse strand
        if self.double:
            for i in hit:
                fwd[i] = rev[i] ^ 3

        # 7. inflow: first all refills for k = 0 .. F-1 (a slot that is dead and unclaimed after step 4; when
        # several k select the same slot the largest k wins), then the births of the refilled slots that are
        # expressible, in increasing slot order [13: Inflow order]
        refilled = set()
        m = MASK64
        for k in range(self.F):
            d = rnd(seed, t, k, 3) % S
            if terrain is not None and (terrain[d] == FIRMAMENT or (terrain[d] == LAND and not self.inflow_land)):
                continue                        # docs/17: no inflow into the firmament; on day 3 only into the sea
            if not alive[d] and claimed_by[d] < 0:
                base = d * G
                x = ((t + 1) * C1 + k * C2 + 4 * C3) & m         # mix(t, k, 4 + j), j = 0, then + C3 per cell
                for j in range(G):
                    z = ((seed ^ x) + C1) & m
                    z = ((z ^ (z >> 30)) * C2) & m
                    z = ((z ^ (z >> 27)) * C3) & m
                    v = (z ^ (z >> 31)) & 3                      # rnd(t, k, 4 + j) mod 4
                    fwd[base + j] = v
                    rev[base + j] = v ^ 3
                    x = (x + C3) & m
                refilled.add(d)
        for d in sorted(refilled):
            if self.expressible(d):
                self.born(d, 0)
                if d1:
                    E_[d] = self.e_spont
                    ledger[2] += self.e_spont
                self.births += 1
                self.spontaneous += 1
                self.birth_log.append((t, -1, d, None, self.genehash(d), False))

        self.tick_no = t + 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=2_999_999_488,
                    help="cells (multiple of G); default 2,929,687 x 1024 = 2,999,999,488, the S of docs/09 section 9 "
                         "(3e9 itself is not a multiple of 1024) [13: Default N]")
    ap.add_argument("--G", type=int, default=1024, help="slot size in cells")
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--I", type=int, default=16, help="instructions per processor per tick")
    ap.add_argument("--T", type=int, default=64, help="lifetime in ticks")
    ap.add_argument("--mu", type=float, default=0.001, help="mutation probability per copied cell")
    ap.add_argument("--D", default="auto", help="damaged cells per tick; auto = max(1, N / 2^20)")
    ap.add_argument("--F", default="auto", help="refilled dead slots per tick; auto = max(1, S / 2^12)")
    ap.add_argument("--no-double", action="store_true", help="single strand: no repair of damage")
    ap.add_argument("--uniform-code", action="store_true", help="codons 0..20 = the 21 meanings, 21..63 NOP")
    ap.add_argument("--dna2", action="store_true", help="DNA-2 (docs/15): TERM on TTA/TTG (uniform: codon 21), "
                    "DIVIDE re-arms P, COPY/COPYR are NOP without a claimed child")
    ap.add_argument("--blank", action="store_true", help="initial tape all A instead of random soup")
    ap.add_argument("--genome", default=None, help="seed genome as codons, e.g. 'ATG GCT CCT ACT CTT GTT GGT TAA'")
    ap.add_argument("--n-seed", type=int, default=0, help="number of seeded organisms at slots k * (S // n_seed)")
    ap.add_argument("--hash-every", type=int, default=1,
                    help="k > 0: hash the state after every k-th tick ((t + 1) mod k = 0) and after the last tick; "
                         "0 = after the last tick only")
    ap.add_argument("--no-hash", action="store_true", help="never compute the state hash")
    ap.add_argument("--hash-init", action="store_true",
                    help="also hash the initial state of section 6, printed as 'tick -1' [13: Tick labels]")
    ap.add_argument("--census-every", type=int, default=10, help="census after every k ticks and at the end; 0 = end only")
    ap.add_argument("--births", default=None, help="write the birth log (TSV) to this file")
    ap.add_argument("--dump", default=None, help="write the final state bytes of section 8 to this file")
    ap.add_argument("--out", default=None, help="write the final census JSON to this file")
    g3 = ap.add_argument_group("DNA-3 (docs/17): the days of creation; every day implies --dna2")
    g3.add_argument("--days", default="", help="days to switch on, e.g. 1 or 1,2 or 1,2,3 or 1,2,3,4 (day 3 needs 1 and 2, day 4 needs 1 to 3)")
    g3.add_argument("--dna3", action="store_true", help="shorthand for --days 1,2,3")
    g3.add_argument("--seed-slots", default=None, help="explicit seed slots, comma-separated (n_seed of them)")
    g3.add_argument("--seed-terrain", default="any", choices=("any", "sea", "land"),
                    help="day 2/3: seeds advance to the next slot of this terrain")
    g3.add_argument("--day-len", type=int, default=32, help="day 1: ticks per day")
    g3.add_argument("--light-len", type=int, default=None, help="day 1: light ticks at the start of each day (default day-len / 2)")
    g3.add_argument("--lmax", type=int, default=16, help="day 1: light level in the light ticks")
    g3.add_argument("--emax", type=int, default=1024, help="day 1: energy store capacity")
    g3.add_argument("--eshift", type=int, default=4, help="day 1: budget = min(I, E >> eshift) instructions per tick")
    g3.add_argument("--e-seed", type=int, default=512, help="day 1: energy of seeded organisms")
    g3.add_argument("--e-spont", type=int, default=128, help="day 1: energy of spontaneous (inflow) organisms")
    g3.add_argument("--g-pass", type=int, default=16, help="day 1 without day 3: passive light gain in sixteenths")
    g3.add_argument("--W", type=int, default=0, help="day 2: grid width in slots (R = S / W rows)")
    g3.add_argument("--firm", type=int, default=1, help="day 2: firmament rows per band (two bands; 0 = none)")
    g3.add_argument("--gp-sea", type=int, default=8, help="day 3: passive gain on sea, sixteenths")
    g3.add_argument("--gp-land", type=int, default=0, help="day 3: passive gain on land, sixteenths")
    g3.add_argument("--gh-sea", type=int, default=16, help="day 3: HARVEST gain on sea, sixteenths")
    g3.add_argument("--gh-land", type=int, default=32, help="day 3: HARVEST gain on land, sixteenths")
    g3.add_argument("--c-copy", type=int, default=1, help="day 3: energy per copied cell")
    g3.add_argument("--maint", type=int, default=1, help="day 3: maintenance per tick")
    g3.add_argument("--inflow-land", action="store_true", help="day 3: inflow also refills land slots")
    g3.add_argument("--c-move", type=int, default=2, help="day 5: energy for a one-slot move (swimming)")
    g3.add_argument("--c-fly", type=int, default=8, help="day 5: energy per slot of a flight (2 to 4 slots)")
    g3.add_argument("--eat-shift", type=int, default=1, help="day 5: an eater takes floor(prey store / 2^eat-shift)")
    g3.add_argument("--eat-eff", type=int, default=8, help="day 5: the eater keeps eat-eff / 16 of what it takes")
    g3.add_argument("--c-walk", type=int, default=4, help="day 6: energy for a one-slot move that touches land")
    g3.add_argument("--task-bonus", type=int, default=8, help="day 6: task bonus unit (x^1, x+1: 1 unit; x^3, x-1: 4 units)")
    g3.add_argument("--shade", action="store_true", help="day 3 option (docs/17 section 14): light gains times (9 - live neighbours) / 9")
    g3.add_argument("--land-cell", type=int, default=32, help="day 3: terrain noise lattice spacing in slots")
    g3.add_argument("--land-thr", type=int, default=32768, help="day 3: land where the noise >= land-thr / 65536")
    g3.add_argument("--year-days", type=int, default=16, help="day 4: days per year")
    g3.add_argument("--moon-days", type=int, default=8, help="day 4: days per lunar cycle")
    g3.add_argument("--season-amp", type=int, default=None, help="day 4: day-length amplitude in ticks (default day-len / 4)")
    g3.add_argument("--lat-amp", type=int, default=8, help="day 4: light lost at the latitude farthest from the sun, sixteenths")
    g3.add_argument("--lmoon", type=int, default=1, help="day 4: moonlight per moon level")
    a = ap.parse_args(argv)
    days = (1, 2, 3) if a.dna3 else tuple(int(x) for x in a.days.replace(",", " ").split())
    seed_slots = [int(x) for x in a.seed_slots.replace(",", " ").split()] if a.seed_slots else None

    genome = parse_genome(a.genome) if a.genome else None
    D, F = parse_auto(a.D), parse_auto(a.F)
    if (D is not None and D < 0) or (F is not None and F < 0):
        ap.error("--D and --F must be 'auto' or a non-negative integer")                # [13: Limits]
    if a.ticks < 1:
        ap.error("--ticks must be at least 1")                                          # [13: Limits]
    if a.N > (1 << 24):
        print("warning: N = %d is far beyond what this byte-per-cell Python reference is meant for" % a.N,
              file=sys.stderr)
    t0 = time.perf_counter()
    w = World(a.N, a.G, I=a.I, T=a.T, mu=a.mu, D=D, F=F, double=not a.no_double,
              uniform=a.uniform_code, blank=a.blank, seed=a.seed, genome=genome, n_seed=a.n_seed, dna2=a.dna2,
              days=days, seed_slots=seed_slots, seed_terrain=a.seed_terrain, W=a.W, firm=a.firm, day_len=a.day_len,
              light_len=a.light_len, lmax=a.lmax, emax=a.emax, eshift=a.eshift, e_seed=a.e_seed, e_spont=a.e_spont,
              g_pass=a.g_pass, gp_sea=a.gp_sea, gp_land=a.gp_land, gh_sea=a.gh_sea, gh_land=a.gh_land,
              c_copy=a.c_copy, maint=a.maint, inflow_land=a.inflow_land, land_cell=a.land_cell, land_thr=a.land_thr,
              year_days=a.year_days, moon_days=a.moon_days, season_amp=a.season_amp, lat_amp=a.lat_amp, lmoon=a.lmoon,
              shade=a.shade, c_move=a.c_move, c_fly=a.c_fly, eat_shift=a.eat_shift, eat_eff=a.eat_eff,
              c_walk=a.c_walk, task_bonus=a.task_bonus)
    t_init = time.perf_counter() - t0
    params = {"N": w.N, "G": w.G, "S": w.S, "ticks": a.ticks, "seed": w.seed, "I": w.I, "T": w.T, "mu": w.mu,
              "D": w.D, "F": w.F, "double": w.double, "code": "uniform" if a.uniform_code else "standard", "dna2": w.dna2,
              "blank": a.blank, "genome": [codon_text(c) for c in genome] if genome else None, "n_seed": a.n_seed,
              "init_secs": round(t_init, 3)}
    if w.days:
        params["days"] = list(w.days)
        if w.d1:
            params.update(day_len=w.day_len, light_len=w.light_len, lmax=w.lmax, emax=w.emax, eshift=w.eshift,
                          e_seed=w.e_seed, e_spont=w.e_spont, gp=list(w.gp))
        if w.d2:
            params.update(W=w.W, R=w.R, firm=w.firm, land=sum(1 for x in w.terrain if x == LAND),
                          firmament=sum(1 for x in w.terrain if x == FIRMAMENT))
        if w.d3:
            params.update(gh=list(w.gh), c_copy=w.c_copy, maint=w.maint, inflow_land=w.inflow_land, shade=w.shade)
        if w.d6:
            params.update(c_walk=w.c_walk, task_bonus=w.task_bonus)
        if w.d5:
            params.update(c_move=w.c_move, c_fly=w.c_fly, eat_shift=w.eat_shift, eat_eff=w.eat_eff)
        if w.d4:
            params.update(year_days=w.year_days, moon_days=w.moon_days, season_amp=w.season_amp, lat_amp=w.lat_amp,
                          lmoon=w.lmoon)
    print("params " + json.dumps(params), file=sys.stderr, flush=True)

    out = sys.stdout
    since = (0, 0, 0, 0)
    last_census = None
    t_hash = 0.0
    # "tick t" labels the state after tick t, t = 0 .. ticks-1; the initial state is "tick -1", printed only with
    # --hash-init [13: Tick labels]. --hash-every k > 0: (t + 1) mod k = 0 and the last tick; 0: last tick only;
    # a negative k (never) is accepted for symmetry with the engine [13: Hash schedule]
    if a.hash_init and not a.no_hash and a.hash_every >= 0:
        th = time.perf_counter()
        out.write("tick -1 hash %016x\n" % w.state_hash())
        t_hash += time.perf_counter() - th
    for t in range(a.ticks):
        w.tick()                                # uses t in rnd
        last = t == a.ticks - 1
        h = None
        if not a.no_hash and (last if a.hash_every == 0 else
                              (a.hash_every > 0 and (last or (t + 1) % a.hash_every == 0))):
            th = time.perf_counter()
            h = w.state_hash()
            t_hash += time.perf_counter() - th
            out.write("tick %d hash %016x\n" % (t, h))
        if last or (a.census_every > 0 and (t + 1) % a.census_every == 0):              # [13: Census schedule]
            last_census = w.census(t, since, h)
            since = (w.births, w.faithful, w.spontaneous, w.deaths)
            out.write("census " + json.dumps(last_census) + "\n")
        out.flush()
    secs = time.perf_counter() - t0

    if a.out and last_census is not None:
        with open(a.out, "w") as f:
            json.dump(last_census, f, indent=1)
            f.write("\n")
    if a.births:                                                                        # [13: Births file]
        # the log is already in the required order (step 3 appends DIVIDE births by increasing parent slot, step 7
        # the spontaneous births by increasing slot); a spontaneous birth has parent -1 and parent hash "-"
        with open(a.births, "w") as f:
            f.write("#tick\tparent\tchild\tparent_hash\tchild_hash\tfaithful\n")
            for (t, p, ch, ph, chh, fl) in w.birth_log:
                f.write("%d\t%d\t%d\t%s\t%016x\t%d\n" % (t, p, ch, "-" if ph is None else "%016x" % ph, chh, fl))
    if a.dump:
        with open(a.dump, "wb") as f:
            f.write(w.state_bytes())
    divide_births = w.births - w.spontaneous
    print("done ticks=%d live=%d births=%d divide_births=%d faithful=%d spontaneous=%d deaths=%d "
          "claims=%d copied_cells=%d secs=%.2f hash_secs=%.2f init_secs=%.2f"
          % (a.ticks, len(w.live), w.births, divide_births, w.faithful, w.spontaneous, w.deaths, w.claims,
             w.copied_cells, secs, t_hash, t_init), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
