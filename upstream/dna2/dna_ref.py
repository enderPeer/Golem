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
    COPYR, REWIND, TERM = range(23)                       # TERM exists only under --dna2 (docs/15)
OP_NAMES = ["NOP", "START", "STOP", "COPY", "R1", "L1", "JNZB", "SETP", "SETQ", "ALLOC", "DIVIDE", "SKNZ", "SKZ",
            "LDA", "STA", "INC", "FLIP", "R3", "L3", "SEEK", "COPYR", "REWIND", "TERM"]
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
                 seed=1, genome=None, n_seed=0, dna2=False):
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
        self.dna2 = dna2
        if dna2:
            self.table = dna2_table(self.table, uniform)
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
            for k in range(n_seed):
                s = k * step
                base = s * G
                for ci in range(self.maxc):
                    c = genome[ci] if ci < len(genome) else self.stop_codon
                    for off, v in ((0, c >> 4), (1, (c >> 2) & 3), (2, c & 3)):
                        fwd[base + 3 * ci + off] = v
                        self.rev[base + 3 * ci + off] = v ^ 3
                # the G mod 3 trailing cells of the slot are not codons and keep their soup content [13: Seeding]
                self.born(s, 0)                 # unconditional, not logged, not counted [13: Seeding, Counters]

    # -- helpers ---------------------------------------------------------------------------------------------
    def born(self, s, depth):
        """A new processor in slot s with the initial values of section 4."""
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
        """Section 8: packed fwd, packed rev, per live processor (slot order) ten LE u64 fields, four counters."""
        out = bytearray(pack2(self.fwd))
        out += pack2(self.rev)
        pk = struct.Struct("<10Q").pack
        for s in sorted(self.live):
            out += pk(s, self.pc[s], self.H[s], self.P[s], self.Q[s], self.child[s] & MASK64,   # -1 -> 2^64 - 1
                      self.A[s], self.age[s], self.depth[s], self.copied[s])
        out += struct.pack("<4Q", self.births, self.faithful, self.spontaneous, self.deaths)
        return bytes(out)

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
        return {
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
            "hash": None if state_hash is None else "%016x" % state_hash,
        }

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
        win = 3 * G + 2                         # SEEK window: candidates H+1 .. H+3G, each needs 3 cells
        alloc_req = []                          # requesting slots, increasing
        marked = []                             # (parent, child) pairs marked by DIVIDE, increasing parent

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
                requested = False
                for _ in range(I):
                    cb = base + 3 * pc
                    op = xtable[fwd[cb] * 16 + fwd[cb + 1] * 4 + fwd[cb + 2]]
                    pc += 1
                    if op == COPY or op == COPYR:
                        if dna2 and child < 0:
                            pass                # docs/15 fix 3: no claimed child, nothing happens (P and Q stay)
                        else:
                          v = fwd[P] if (base <= P < own_hi or ch_lo <= P < ch_hi) else fwd_old[P]
                          if op == COPYR:
                              v ^= 3
                          if ch_lo <= Q < ch_hi:  # child claimed and Q inside it: write
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
                            L = gene_len(fwd, is_stop, base, maxc)                      # [13: Degenerate genes]
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
                            tb = base + 3 * pc
                            pat = bytes((fwd[tb] ^ 3, fwd[tb + 1] ^ 3, fwd[tb + 2] ^ 3))
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
                    # NOP: nothing
                    if pc >= L:
                        pc = 1
                pc_[s] = pc; H_[s] = H; P_[s] = P; Q_[s] = Q; child_[s] = child; A_[s] = A; copied_[s] = copied

        # 2. resolve ALLOC: increasing slot order, so the lowest requester takes a contested candidate
        Sm1 = S - 1
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
                ph = self.genehash(p)           # the parent's gene as it is now, after its own writes of the tick
                chh = self.genehash(ch)
                f = ph == chh
                self.births += 1                                                        # [13: Counters]
                self.faithful += f
                self.birth_log.append((t, p, ch, ph, chh, f))

        # 4. deaths (age += 1 for every processor alive now, this tick's newborns included) [13: Age]
        T = self.T
        age_ = self.age
        for s in sorted(live):
            age_[s] += 1
            if age_[s] >= T or not self.expressible(s):
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
    a = ap.parse_args(argv)

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
              uniform=a.uniform_code, blank=a.blank, seed=a.seed, genome=genome, n_seed=a.n_seed, dna2=a.dna2)
    t_init = time.perf_counter() - t0
    params = {"N": w.N, "G": w.G, "S": w.S, "ticks": a.ticks, "seed": w.seed, "I": w.I, "T": w.T, "mu": w.mu,
              "D": w.D, "F": w.F, "double": w.double, "code": "uniform" if a.uniform_code else "standard", "dna2": a.dna2,
              "blank": a.blank, "genome": [codon_text(c) for c in genome] if genome else None, "n_seed": a.n_seed,
              "init_secs": round(t_init, 3)}
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
