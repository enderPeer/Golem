"""One verified DNA-3 compute slice, not an arbitrary genotype compiler.

The observed O6 organism remains an ecological looping program. This bridge
extracts its actual IN -> INC -> LDA -> OUT data path and gives that slice a
new, explicit argument/return ABI. An untouched frozen World.tick validates
the slice under bounded fixtures. Only its external IN source is injected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types


SCHEMA = 'golem-dna3-compute-slice-v1'
FROZEN = Path(__file__).resolve().parents[1] / 'upstream' / 'dna3'
PINNED = {
    'dna_ref.py': '494da6d7a876c17b4df12474c270a88fcb5d0f0f9b2a7e84fc0244a7b0cc9ceb',
    'dna3_ckpt_genes.py': 'b301f6a0b5ea58cc1401a1581d18ff86d288b562a1ad1bfce44bc51e8038fa39',
    'taskers_t102400.json': 'cd4731488dafc12adb140f7efbf2a640a72be3b8ada8083cfb8d79aef430d8d9',
    'LICENSE': '5cdbc44e43c4d3cf4988d3154b0a7b0b9715ad33bb87fec5a9a87b958314055a',
    'provenance.json': '1fcc0698d362ed88802d83febd1fa0a63060c1d452c4cd707568121f20d5b980',
}
SPELLING = 'ATG AGG GCT CTT TTG GGG TAT AGT AGG AGA TCG TCA CAA'
OPS = 'START HARVEST ALLOC COPY TERM DIVIDE LDA SENSE HARVEST HARVEST OUT IN INC'
SLICE_PCS = (11, 12, 6, 10)  # Zero-based codon indices, with START at zero.
SLICE_OPS = ('IN', 'INC', 'LDA', 'OUT')
CONTEXTS = (
    {'name': 'sea_day_blank', 'seed': 7, 'start_tick': 0, 'blank': True,
     'land_thr': 65536, 'shade': False, 'maint': 0},
    {'name': 'land_night_soup', 'seed': 73, 'start_tick': 31, 'blank': False,
     'land_thr': 0, 'shade': True, 'maint': 1},
    {'name': 'season_change_soup', 'seed': 97, 'start_tick': 256, 'blank': False,
     'land_thr': 32768, 'shade': True, 'maint': 1},
)
ABI = {
    'name': 'dna3-u2-argument-slice-v1', 'domain': [0, 1, 2, 3],
    'input': 'Replace only parent slot 0 rnd(seed,tick,0,4000) with supplied two-bit argument; IN still writes both strands, H, A, xin and open in the frozen VM.',
    'output': 'The accumulator at the next actual parent OUT with open=1; OUT is executed and must consume open.',
    'slice_reset': 'Each standalone call starts fresh local cell/accumulator state; IN overwrites both with x.',
    'reference_reset': 'Fresh 4x4-slot World per fixture, G=256, one seeded observed organism, no mutation/inflow/damage. Repeated fixtures keep the same organism between calls.',
    'tick_policy': 'I=1, one codon per tick, to observe committed instruction state without changing World.tick.',
    'energy': 'emax=e_seed=4096, eshift=0; unchanged ecological costs and sky rules otherwise, with fixture-specific maintenance/shading.',
    'reference_parameters': {'N': 4096, 'G': 256, 'I': 1, 'T': 0, 'mu': 0,
        'D': 0, 'F': 0, 'double': True, 'n_seed': 1, 'seed_slots': [0],
        'days': [1, 2, 3, 4, 5, 6], 'W': 4, 'firm': 0, 'land_cell': 1,
        'emax': 4096, 'e_seed': 4096, 'e_spont': 0, 'eshift': 0},
    'reference_context_parameters': ['seed', 'blank', 'land_thr', 'maint', 'shade', 'start_tick'],
    'other_reference_parameters': 'Defaults in the byte-pinned World constructor; genome is the recorded observed spelling. T=0 disables organism age expiry in these fixtures.',
    'bounds': {'max_calls': 64, 'max_ticks_per_fixture': 2048, 'slots': 16, 'cells_per_slot': 256},
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _argument(value):
    if type(value) is not int or not 0 <= value <= 3:
        raise ValueError('increment_mod4 accepts only integer values 0..3 (not bool)')
    return value


def _sources():
    contents = {}
    for name, expected in PINNED.items():
        path = FROZEN / name
        if path.is_symlink():
            raise ValueError('frozen DNA-3 source must not be a symlink')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('frozen DNA-3 source hash mismatch: ' + name)
        contents[name] = content
    return contents


def _reference(contents):
    # This exact MIT source is pinned above; importing it runs no CLI/main code,
    # creates no bytecode and touches no original upstream file or workload.
    module = types.ModuleType('_golem_frozen_dna3_reference')
    exec(compile(contents['dna_ref.py'], str(FROZEN / 'dna_ref.py'), 'exec'), module.__dict__)
    return module


def _observed_candidate(contents, reference):
    extraction = json.loads(contents['taskers_t102400.json'])
    row = extraction['top'][0]
    if (extraction['tick'] != 102400 or row['spelling'] != SPELLING or row['ops'] != OPS or
            row['count'] != 42577 or row['spelling_count'] != 9846):
        raise ValueError('unexpected observed O6 candidate')
    table = reference.dna2_table(reference.standard_table(), False)
    for transform in (reference.dna3_table, reference.dna3_day4_table,
                      reference.dna3_day5_table, reference.dna3_day6_table):
        table = transform(table, False)
    genome = [reference.codon_value(codon) for codon in row['spelling'].split()]
    decoded = [reference.OP_NAMES[table[codon]] for codon in genome]
    if ' '.join(decoded) != row['ops'] or tuple(decoded[pc] for pc in SLICE_PCS) != SLICE_OPS:
        raise ValueError('observed spelling does not decode to the declared compute slice')
    return genome, {'world': 'O6_day6', 'extraction_tick': extraction['tick'], 'row': 0,
                   'spelling': row['spelling'], 'ops': row['ops'], 'codons': len(genome),
                   'operation_sequence_carriers': row['count'], 'spelling_carriers': row['spelling_count'],
                   'evidence_scope': 'Frozen upstream checkpoint-extraction JSON; original large checkpoint was not re-extracted here.',
                   'source_pcs_zero_based': list(SLICE_PCS), 'slice_ops': list(SLICE_OPS)}


def execute_slice(x, operations=SLICE_OPS):
    """Execute the fixed slice's small cell/accumulator machine, with no tables."""
    _argument(x)
    if tuple(operations) != SLICE_OPS:
        raise ValueError('only the verified observed IN/INC/LDA/OUT slice is accepted')
    cell = accumulator = None
    opened = False
    for operation in operations:
        if operation == 'IN':
            cell = accumulator = x
            opened = True
        elif operation == 'INC':
            cell = (cell + 1) & 3
        elif operation == 'LDA':
            accumulator = cell
        elif operation == 'OUT' and opened:
            return accumulator
    raise ValueError('slice did not produce its one answer')


def _full_reference_fixture(reference, genome, inputs, context):
    """Run actual ecological code and certify its observed computational slice."""
    if not inputs or len(inputs) > 64:
        raise ValueError('reference fixture needs 1..64 inputs')
    inputs = [_argument(x) for x in inputs]
    world = reference.World(4096, 256, I=1, T=0, mu=0, D=0, F=0, double=True,
        blank=context['blank'], seed=context['seed'], genome=genome, n_seed=1, seed_slots=[0],
        days=(1, 2, 3, 4, 5, 6), W=4, firm=0, land_cell=1, land_thr=context['land_thr'],
        emax=4096, e_seed=4096, e_spont=0, eshift=0, maint=context['maint'], shade=context['shade'])
    world.tick_no = context['start_tick']
    original_gene = world.genehash(0)
    original_rnd = reference.rnd
    fed, answers, path = [], [], []
    started = False
    divides = 0

    def supplied_input(seed, tick, slot, stream):
        if slot == 0 and stream == 4000:
            if len(fed) >= len(inputs) or world.pc[0] != SLICE_PCS[0]:
                raise ValueError('unexpected or unbounded parent IN request')
            value = inputs[len(fed)]
            fed.append(value)
            return value
        return original_rnd(seed, tick, slot, stream)

    reference.rnd = supplied_input
    try:
        for instruction in range(2048):
            if not world.alive[0] or world.xs[0] != 0 or world.E[0] < 2:
                raise ValueError('parent died, changed execution source or exhausted the bounded energy fixture')
            pc = world.pc[0]
            codon = world.fwd[3 * pc] * 16 + world.fwd[3 * pc + 1] * 4 + world.fwd[3 * pc + 2]
            operation = reference.OP_NAMES[world.xtable[codon]]
            before = {'A': world.A[0], 'H': world.H[0], 'cell': world.fwd[255],
                      'open': world.tmask[0], 'xin': world.xin[0]}
            if operation == 'IN':
                if started:
                    raise ValueError('a second IN appeared before the pending answer')
                started, path = True, []
            if started:
                path.append([pc, operation])
            if operation == 'DIVIDE' and world.child[0] >= 0 and world.copied[0] >= 6:
                divides += 1
            # No interpreter opcode, scheduler, reproduction or task-reward code
            # is replaced. Only the input-source hook above supplies arguments.
            world.tick()
            after = {'A': world.A[0], 'H': world.H[0], 'cell': world.fwd[255],
                     'open': world.tmask[0], 'xin': world.xin[0]}
            if world.genehash(0) != original_gene:
                raise ValueError('full gene mutated during a supposedly fixed-program fixture')
            if started:
                if operation == 'IN':
                    value = fed[-1]
                    if after != {'A': value, 'H': 255, 'cell': value, 'open': 1, 'xin': value}:
                        raise ValueError('IN did not preserve the declared reference input semantics')
                elif operation == 'INC':
                    if before['H'] != 255 or after != {**before, 'cell': (before['cell'] + 1) & 3}:
                        raise ValueError('INC writes outside the declared compute cell')
                elif operation == 'LDA':
                    if before['H'] != 255 or after != {**before, 'A': before['cell']}:
                        raise ValueError('LDA depends on undeclared memory')
                elif operation == 'OUT':
                    if before['open'] != 1 or after != {**before, 'open': 0}:
                        raise ValueError('OUT did not consume exactly the pending answer')
                    projection = [position for position, op in path if op in SLICE_OPS]
                    if projection != list(SLICE_PCS):
                        raise ValueError('actual execution does not match the proposed noncontiguous slice')
                    expected = execute_slice(before['xin'])
                    if before['A'] != expected:
                        raise ValueError('full ecological reference differs from the extracted slice')
                    answers.append({'input': before['xin'], 'output': before['A'], 'path': list(path)})
                    started = False
                elif operation not in ('HARVEST', 'ALLOC', 'COPY', 'TERM', 'DIVIDE', 'SENSE') or after != before:
                    raise ValueError('ecological instruction changed computational state or used an unapproved dependency')
                if world.rev[255] != (world.fwd[255] ^ 3):
                    raise ValueError('input cell complement strand was not preserved')
            if len(answers) == len(inputs):
                return {'context': dict(context), 'instructions': instruction + 1, 'inputs': fed,
                        'answers': answers, 'successful_parent_divisions': divides,
                        'parent_gene_unchanged': True, 'discarded_operations_preserve_A_H_io_xin_open': True}
        raise ValueError('reference fixture exceeded its instruction budget')
    finally:
        reference.rnd = original_rnd


def _export_source(program_id, operations):
    if tuple(operations) != SLICE_OPS:
        raise ValueError('export requires the verified fixed slice')
    statements = {'IN': ['cell = x', 'accumulator = x'], 'INC': ['cell = (cell + 1) & 3'],
                  'LDA': ['accumulator = cell'], 'OUT': ['return accumulator']}
    lines = ['# MIT; source and full copyright notice: upstream/dna3/LICENSE',
             '# Source-derived compute slice, not the complete ecological organism.',
             '# Verified program: ' + program_id,
             'def increment_mod4(x):',
             '    """Return the verified two-bit slice result; integer domain 0..3."""',
             '    if type(x) is not int or not 0 <= x <= 3:',
             '        raise ValueError("increment_mod4 requires an integer in 0..3")']
    for pc, operation in zip(SLICE_PCS, operations):
        lines.append('    # Original zero-based DNA-3 PC %d: %s' % (pc, operation))
        lines.extend('    ' + statement for statement in statements[operation])
    return '\n'.join(lines) + '\n'


def verify_increment_bridge():
    """Recompute the frozen proof; no source writes, remote jobs or model calls."""
    contents = _sources()
    reference = _reference(contents)
    genome, origin = _observed_candidate(contents, reference)
    reset_cases, repeated_cases = [], []
    for context in CONTEXTS:
        reset_cases.extend(_full_reference_fixture(reference, genome, [x], context) for x in range(4))
        repeated_cases.append(_full_reference_fixture(reference, genome, [0, 1, 2, 3] * 16, context))
    table = [case['answers'][0]['output'] for case in reset_cases[:4]]
    if table != [(x + 1) & 3 for x in range(4)]:
        raise ValueError('observed verified function is not increment modulo four')
    if not all(case['successful_parent_divisions'] >= 1 for case in repeated_cases):
        raise ValueError('repeated-call proof never exercised the parent division transition')
    sense_paths = {tuple(position for position, _ in answer['path'])
                   for case in repeated_cases for answer in case['answers']}
    if not any(9 in path for path in sense_paths) or not any(9 not in path for path in sense_paths):
        raise ValueError('reference fixtures did not cover both SENSE-controlled harvest branches')
    program_id = 'dna3-o6-incmod4-slice-v1-' + _digest({'origin': origin, 'source': PINNED, 'abi': ABI})[:20]
    source = _export_source(program_id, origin['slice_ops'])
    # Execute only this fixed generated template, independently from the slice
    # interpreter and full World.tick; it has no imports or external effects.
    namespace = {}
    exec(compile(source, '<verified DNA-3 slice export>', 'exec'), namespace)
    exported = [namespace['increment_mod4'](x) for x in range(4)]
    if exported != table:
        raise ValueError('executable export differs from the reference-verified truth table')
    certificate = {'schema': SCHEMA, 'program_id': program_id, 'task': 'increment_mod4',
        'domain': [0, 1, 2, 3], 'truth_table': table, 'origin': origin, 'abi': ABI,
        'source_files_sha256': dict(PINNED), 'module_file': 'increment_mod4.py',
        'module_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'verification': {'reference': 'Exact frozen World.tick with explicit parent IN-source adapter',
            'reset_cases': reset_cases, 'repeated_cases': repeated_cases,
            'verified_answers': sum(len(case['answers']) for case in reset_cases + repeated_cases),
            'export_outputs': exported, 'both_sense_branches_exercised': True,
            'parent_division_exercised': True},
        'limitations': ['A source-derived noncontiguous compute slice; not an arbitrary genotype compiler.',
            'The argument/return ABI replaces the ecological random input source and isolates computation.',
            'Proof covers all four inputs under the recorded bounded fixtures, not every ecological state.',
            'No minimality, hardware speedup, general addition, sorting, evolved PRNG, or conversational-quality claim.',
            'Evolutionary origin does not establish superiority to a hand-written increment function.']}
    certificate['certificate_sha256'] = _digest(certificate)
    return certificate


def build_increment_bridge(output_dir):
    """Write a deterministic verified certificate, module and source license."""
    directory = Path(output_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError('bridge output directory must be new or empty')
    certificate = verify_increment_bridge()
    directory.mkdir(parents=True, exist_ok=True)
    source = _export_source(certificate['program_id'], certificate['origin']['slice_ops'])
    (directory / certificate['module_file']).write_text(source, encoding='utf-8', newline='\n')
    (directory / 'LICENSE').write_bytes(_sources()['LICENSE'])
    path = directory / 'certificate.json'
    path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    return {**certificate, 'certificate_path': str(path.resolve()),
            'module_path': str((directory / certificate['module_file']).resolve())}


def load_verified_bridge(path_or_dir):
    """Reject tampering and recompute the bounded proof before downstream use."""
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / 'certificate.json'
    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('unsafe or oversized bridge certificate')
    candidate = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(candidate, dict) or candidate.get('schema') != SCHEMA:
        raise ValueError('unsupported bridge certificate')
    material = {key: value for key, value in candidate.items() if key != 'certificate_sha256'}
    if candidate.get('certificate_sha256') != _digest(material):
        raise ValueError('bridge certificate hash mismatch')
    verified = verify_increment_bridge()
    if candidate != verified:
        raise ValueError('bridge certificate differs from the recomputed source proof')
    module = path.parent / 'increment_mod4.py'
    expected = _export_source(verified['program_id'], verified['origin']['slice_ops']).encode()
    if module.is_symlink() or module.read_bytes() != expected:
        raise ValueError('bridge executable export differs from verified source')
    if (path.parent / 'LICENSE').read_bytes() != _sources()['LICENSE']:
        raise ValueError('bridge source license missing or changed')
    return verified
