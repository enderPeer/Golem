"""Sense-preserving WordNet graph, sparse connectivity rank, and typed queries.

Synsets are concepts. Co-membership defines synonyms ONLY within that concept.
Lexical pointers keep their specific source and target lemmas. PageRank measures
directed graph connectivity, not importance, frequency, truth or synonymy.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import heapq
import io
import json
import math
import os
from pathlib import Path
import re
import zipfile

from .data import _atomic_json, _sha256, fetch_file

SCHEMA = 'golem-wordnet-graph-v1'
SOURCE_REVISION = '96f9b3252457a2b97e52aec64c3dfceeb5c312d5'
SOURCE_URL = f'https://raw.githubusercontent.com/nltk/nltk_data/{SOURCE_REVISION}/packages/corpora/wordnet.zip'
SOURCE_SHA256 = 'cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59'
RELATIONS = {'!': 'antonym', '@': 'hypernym', '@i': 'instance_hypernym', '~': 'hyponym',
             '~i': 'instance_hyponym', '#m': 'member_holonym', '#s': 'substance_holonym',
             '#p': 'part_holonym', '%m': 'member_meronym', '%s': 'substance_meronym',
             '%p': 'part_meronym', '=': 'attribute', '+': 'derivationally_related',
             ';c': 'domain_topic', '-c': 'member_topic', ';r': 'domain_region',
             '-r': 'member_region', ';u': 'domain_usage', '-u': 'member_usage',
             '*': 'entailment', '>': 'cause', '^': 'also_see', '$': 'verb_group',
             '&': 'similar_to', '<': 'participle_of_verb', '\\': 'pertainym'}
_ADJECTIVE_MARKER = re.compile(r'\((?:a|p|ip)\)$')


def normalize_word(word: str) -> str:
    return ' '.join(word.replace('_', ' ').split()).casefold()


def _node_id(pos: str, offset: str) -> str:
    if pos not in ('n', 'v', 'a', 's', 'r'):
        raise ValueError('invalid WordNet part of speech')
    return ('a' if pos == 's' else pos) + ':' + f'{int(offset):08d}'


def parse_data_line(line: str) -> tuple[str, dict, list[dict]] | None:
    """Parse the official data.* format; resolve lexical targets after all files."""
    if not line or line[0].isspace():
        return None
    fields, separator, gloss = line.rstrip('\r\n').partition('|')
    values = fields.split()
    if not separator or len(values) < 5:
        raise ValueError('invalid WordNet synset record')
    offset, lex_file, pos = values[:3]
    node_id = _node_id(pos, offset)
    count = int(values[3], 16)
    cursor, lemmas, lex_ids, original_forms, markers = 4, [], [], [], []
    for _ in range(count):
        form, lex_id = values[cursor:cursor + 2]
        cursor += 2
        marker = _ADJECTIVE_MARKER.search(form) if pos in ('a', 's') else None
        clean = form[:marker.start()] if marker else form
        original_forms.append(clean)
        lemmas.append(clean.replace('_', ' '))
        lex_ids.append(int(lex_id, 16))
        markers.append(marker.group(0)[1:-1] if marker else None)
    pointer_count = int(values[cursor])
    cursor += 1
    edges = []
    for _ in range(pointer_count):
        symbol, target_offset, target_pos, source_target = values[cursor:cursor + 4]
        cursor += 4
        if symbol not in RELATIONS or len(source_target) != 4:
            raise ValueError('unknown WordNet pointer or invalid lexical index field')
        source_index, target_index = int(source_target[:2], 16), int(source_target[2:], 16)
        if bool(source_index) != bool(target_index):
            raise ValueError('lexical pointers must specify both word indexes')
        relation = 'derived_from_adjective' if symbol == '\\' and pos == 'r' else RELATIONS[symbol]
        edge = {'source': node_id, 'target': _node_id(target_pos, target_offset), 'relation': relation}
        if source_index:
            if source_index > len(lemmas):
                raise ValueError('source lexical pointer is outside its synset')
            edge.update(source_lemma=lemmas[source_index - 1], source_index=source_index,
                        target_index=target_index)
        edges.append(edge)
    frames = []
    if pos == 'v':
        frame_count = int(values[cursor])
        cursor += 1
        for _ in range(frame_count):
            plus, frame, word_index = values[cursor:cursor + 3]
            cursor += 3
            if plus != '+':
                raise ValueError('invalid verb frame')
            frames.append({'frame': int(frame), 'word_index': int(word_index, 16)})
    if cursor != len(values):
        raise ValueError('unparsed fields in WordNet synset record')
    node = {'lemmas': lemmas, 'pos': pos, 'definition': gloss.strip(), 'lex_ids': lex_ids,
            'lex_file': int(lex_file), 'lemma_forms': original_forms}
    if any(markers):
        node['syntactic_markers'] = markers
    if frames:
        node['verb_frames'] = frames
    return node_id, node, edges


def parse_wordnet(archive: str | Path) -> tuple[dict, list[dict], bytes, dict, dict]:
    nodes, edges, source_files, sense_order = {}, [], {}, defaultdict(list)
    with zipfile.ZipFile(archive) as source:
        for suffix in ('noun', 'verb', 'adj', 'adv'):
            name = 'wordnet/data.' + suffix
            info = source.getinfo(name)
            if info.file_size > 64 * 1024 * 1024:
                raise ValueError('WordNet data member exceeds size limit')
            raw = source.read(name)
            source_files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            for line in io.StringIO(raw.decode('ascii')):
                parsed = parse_data_line(line)
                if parsed is None:
                    continue
                identity, node, pointers = parsed
                if identity in nodes:
                    raise ValueError('duplicate WordNet synset')
                nodes[identity] = node
                edges.extend(pointers)
            index_name = 'wordnet/index.' + suffix
            if source.getinfo(index_name).file_size > 32 * 1024 * 1024:
                raise ValueError('WordNet index member exceeds size limit')
            index_bytes = source.read(index_name)
            source_files[index_name] = {'bytes': len(index_bytes), 'sha256': hashlib.sha256(index_bytes).hexdigest()}
            for line in io.StringIO(index_bytes.decode('ascii')):
                if not line or line[0].isspace():
                    continue
                values = line.split()
                lemma, pos, count, pointer_count = values[:4]
                first_offset = 6 + int(pointer_count)
                offsets = values[first_offset:first_offset + int(count)]
                if len(offsets) != int(count):
                    raise ValueError('invalid WordNet index sense count')
                sense_order[normalize_word(lemma)].extend(_node_id(pos, offset) for offset in offsets)
        if source.getinfo('wordnet/LICENSE').file_size > 128 * 1024:
            raise ValueError('WordNet license exceeds size limit')
        license_bytes = source.read('wordnet/LICENSE')
    for edge in edges:
        if edge['target'] not in nodes:
            raise ValueError('WordNet pointer references a missing synset')
        if 'target_index' in edge:
            words = nodes[edge['target']]['lemmas']
            if edge['target_index'] > len(words):
                raise ValueError('target lexical pointer is outside its synset')
            edge['target_lemma'] = words[edge['target_index'] - 1]
    return nodes, edges, license_bytes, source_files, dict(sense_order)


def pagerank(nodes, edges: list[dict], iterations: int = 30, damping: float = .85) -> dict[str, float]:
    """Sparse directed unit-link PageRank; subset inputs ignore external edges.

    Multiple typed/lexical edges between the same two synsets count once for this
    global connectivity measure. Dangling mass is redistributed uniformly.
    """
    if type(iterations) is not int or not 1 <= iterations <= 200:
        raise ValueError('iterations must be an integer in 1..200')
    if not math.isfinite(damping) or not 0 <= damping < 1:
        raise ValueError('damping must be finite and in [0,1)')
    identities = sorted(nodes)
    if not identities:
        return {}
    index = {identity: i for i, identity in enumerate(identities)}
    links = [set() for _ in identities]
    for edge in edges:
        if edge['source'] in index and edge['target'] in index:
            links[index[edge['source']]].add(index[edge['target']])
    links = [sorted(targets) for targets in links]
    n = len(identities)
    rank = [1. / n] * n
    for _ in range(iterations):
        dangling = sum(rank[i] for i, targets in enumerate(links) if not targets)
        base = (1. - damping + damping * dangling) / n
        updated = [base] * n
        for i, targets in enumerate(links):
            if targets:
                contribution = damping * rank[i] / len(targets)
                for target in targets:
                    updated[target] += contribution
        total = sum(updated)
        rank = [value / total for value in updated]
    return dict(zip(identities, rank))


def _word_index(nodes: dict, sense_order: dict | None = None) -> dict:
    index = defaultdict(list)
    for identity, node in nodes.items():
        for word in sorted({normalize_word(lemma) for lemma in node['lemmas']}):
            index[word].append(identity)
    ordered = {}
    for word, senses in sorted(index.items()):
        requested = (sense_order or {}).get(word, [])
        known = set(senses)
        first = list(dict.fromkeys(identity for identity in requested if identity in known))
        ordered[word] = first + sorted(known - set(first))
    return ordered


def build_graph(output_dir: str | Path, iterations: int = 30, damping: float = .85,
                cache_dir: str | Path | None = None) -> Path:
    output = Path(output_dir)
    graph_path = output / 'graph.json'
    settings = {'iterations': iterations, 'damping': damping}
    if graph_path.exists():
        existing = load_graph(graph_path)
        if existing['rank']['settings'] != settings:
            raise ValueError('existing graph has different ranking settings; choose a new directory')
        return graph_path
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory is occupied without a verified graph')
    cache = Path(cache_dir) if cache_dir else output.parent / '.lexicon-cache'
    archive, metadata = fetch_file(SOURCE_URL, cache, expected_sha256=SOURCE_SHA256, max_bytes=64 * 1024 * 1024)
    nodes, edges, license_bytes, source_files, sense_order = parse_wordnet(archive)
    synset_rank = pagerank(nodes, edges, iterations=iterations, damping=damping)
    index = _word_index(nodes, sense_order)
    word_rank = defaultdict(float)
    for identity, node in nodes.items():
        words = sorted({normalize_word(lemma) for lemma in node['lemmas']})
        for word in words:
            word_rank[word] += synset_rank[identity] / len(words)
    relation_counts = defaultdict(int)
    for edge in edges:
        relation_counts[edge['relation']] += 1
    provenance = {'source': 'Princeton WordNet', 'version': '3.0', 'revision': SOURCE_REVISION,
                  'distribution': 'NLTK official nltk_data repository, pinned archive', 'archive': metadata,
                  'source_files': source_files,
                  'license': {'path': 'WORDNET-LICENSE.txt', 'sha256': hashlib.sha256(license_bytes).hexdigest(),
                              'url': 'https://wordnet.princeton.edu/license-and-commercial-use'},
                  'format_reference': 'https://wordnet.princeton.edu/documentation/wndb5wn'}
    counts = {'synsets': len(nodes), 'words': len(index), 'edges': len(edges),
              'lexical_edges': sum('source_lemma' in edge for edge in edges),
              'relations': dict(sorted(relation_counts.items()))}
    graph = {'schema': SCHEMA, 'provenance': provenance, 'counts': counts,
             'nodes': nodes, 'edges': edges, 'word_index': index,
             'sense_order': 'Official index order within POS; POS groups noun, verb, adjective, adverb. No cross-POS frequency claim.',
             'rank': {'synsets': synset_rank, 'words': dict(sorted(word_rank.items())), 'settings': settings,
                      'method': 'directed unweighted PageRank on unique synset links; uniform teleport and dangling redistribution',
                      'word_aggregation': 'Each synset mass divided among its unique lemmas; each word sums across its senses',
                      'interpretation': 'Connectivity only; polysemy affects word rank. Not importance, frequency or semantic similarity.',
                      'training_warning': 'Full-graph rank is exploratory; recompute on training-only nodes and edges for training.'}}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'WORDNET-LICENSE.txt').write_bytes(license_bytes)
    temporary = output / 'graph.json.tmp'
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(graph, handle, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, graph_path)
    _atomic_json(output / 'manifest.json', {'schema': SCHEMA, 'graph_sha256': _sha256(graph_path),
                                          'settings': settings, 'counts': counts, 'provenance': provenance})
    return graph_path


def load_graph(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / 'graph.json'
    manifest = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA or _sha256(path) != manifest.get('graph_sha256'):
        raise ValueError('lexical graph checksum/schema mismatch')
    graph = json.loads(path.read_text(encoding='utf-8'))
    if graph.get('schema') != SCHEMA or graph.get('provenance') != manifest.get('provenance'):
        raise ValueError('lexical graph provenance/schema mismatch')
    license_record = graph['provenance']['license']
    if license_record.get('path') != 'WORDNET-LICENSE.txt' or _sha256(path.parent / 'WORDNET-LICENSE.txt') != license_record['sha256']:
        raise ValueError('WordNet license checksum mismatch')
    if graph.get('counts') != manifest.get('counts'):
        raise ValueError('lexical graph counts differ from manifest')
    if len(graph['nodes']) != graph['counts']['synsets'] or len(graph['edges']) != graph['counts']['edges']:
        raise ValueError('lexical graph content counts mismatch')
    return graph


_WEIGHTS = {'similar_to': .90, 'hypernym': .80, 'hyponym': .72, 'instance_hypernym': .75,
            'instance_hyponym': .70, 'derivationally_related': .70, 'verb_group': .75,
            'also_see': .60, 'antonym': .55, 'entailment': .65, 'cause': .60,
            'member_meronym': .65, 'substance_meronym': .65, 'part_meronym': .65,
            'member_holonym': .65, 'substance_holonym': .65, 'part_holonym': .65}


def rank_related(graph: dict, word: str, sense: str | None = None, limit: int = 20,
                 max_depth: int = 2) -> list[dict]:
    """Rank sense-specific paths with an explicit bounded heuristic, not embeddings.

    Score = product(relation weights) * .7**(distance-1), with within-sense
    synonyms at 1. PageRank only breaks ties. Antonyms remain labelled opposites.
    Only outgoing declared pointers are followed; no inverse relation is invented.
    """
    if type(limit) is not int or not 1 <= limit <= 200 or not 1 <= max_depth <= 3:
        raise ValueError('limit must be 1..200 and max_depth 1..3')
    normalized = normalize_word(word)
    senses = graph['word_index'].get(normalized, [])
    if sense is not None:
        if sense not in senses:
            raise ValueError('requested sense does not belong to this word')
        senses = [sense]
    adjacency = defaultdict(list)
    for edge in graph['edges']:
        adjacency[edge['source']].append(edge)
    results = []
    rank = graph.get('rank', {}).get('synsets', {})
    for seed in senses:
        seed_node = graph['nodes'][seed]
        synonyms = [lemma for lemma in seed_node['lemmas'] if normalize_word(lemma) != normalized]
        if synonyms:
            results.append({'seed_sense': seed, 'seed_definition': seed_node['definition'], 'target_sense': seed,
                            'lemmas': synonyms, 'definition': seed_node['definition'], 'relation': 'same_sense_synonym',
                            'path': [], 'distance': 0, 'score': 1., 'global_connectivity_rank': rank.get(seed, 0.)})
        queue = [(-1., 0, seed, normalized, ())]
        visited, best = {}, {}
        expanded = 0
        while queue and expanded < 1000:
            negative, distance, source, focused_lemma, relations = heapq.heappop(queue)
            key = (source, focused_lemma)
            score = -negative
            if visited.get(key, -1.) >= score:
                continue
            visited[key] = score
            expanded += 1
            if distance >= max_depth:
                continue
            for edge in adjacency[source]:
                if ('source_lemma' in edge and focused_lemma and
                        normalize_word(edge['source_lemma']) != focused_lemma):
                    continue
                target = edge['target']
                if target == seed:
                    continue
                next_distance = distance + 1
                next_score = score * _WEIGHTS.get(edge['relation'], .50) * (.7 if distance else 1.)
                path = relations + (edge['relation'],)
                next_focus = normalize_word(edge['target_lemma']) if 'target_lemma' in edge else ''
                result_key = (target, next_focus)
                if result_key not in best or next_score > best[result_key]['score']:
                    node = graph['nodes'][target]
                    best[result_key] = {'seed_sense': seed, 'seed_definition': seed_node['definition'],
                        'target_sense': target, 'lemmas': [edge['target_lemma']] if next_focus else node['lemmas'],
                        'definition': node['definition'], 'relation': edge['relation'] if next_distance == 1 else 'related_via_path',
                        'path': list(path), 'distance': next_distance, 'score': next_score,
                        'global_connectivity_rank': rank.get(target, 0.)}
                if next_distance < max_depth:
                    heapq.heappush(queue, (-next_score, next_distance, target, next_focus, path))
        results.extend(best.values())
    return sorted(results, key=lambda row: (-row['score'], -row['global_connectivity_rank'],
                                           row['seed_sense'], row['target_sense'], row['lemmas']))[:limit]


def relation_path(graph: dict, word_a: str, word_b: str, source_sense: str | None = None,
                  target_sense: str | None = None, max_hops: int = 4, max_nodes: int = 10000) -> dict:
    """Bounded shortest declared relation path; never compute an all-pairs matrix.

    Defaults select the source's first indexed sense, then the target's first
    matching-POS sense when possible. Explicit sense arguments override this.
    Lexical pointers stay attached to their exact lemmas; antonyms are reported
    as opposition and never re-labelled synonyms or positive similarity.
    """
    if type(max_hops) is not int or not 0 <= max_hops <= 6 or type(max_nodes) is not int or not 1 <= max_nodes <= 100000:
        raise ValueError('max_hops must be 0..6 and max_nodes 1..100000')
    a, b = normalize_word(word_a), normalize_word(word_b)
    a_senses, b_senses = graph['word_index'].get(a, []), graph['word_index'].get(b, [])
    if not a_senses or not b_senses:
        raise ValueError('both words must occur in the lexical graph')
    if source_sense is not None and source_sense not in a_senses:
        raise ValueError('source sense does not belong to source word')
    if target_sense is not None and target_sense not in b_senses:
        raise ValueError('target sense does not belong to target word')
    source = source_sense or (next((sid for sid in a_senses if target_sense and sid[0] == target_sense[0]), a_senses[0]))
    target = target_sense or next((sid for sid in b_senses if sid[0] == source[0]), b_senses[0])
    result = {'source_word': a, 'target_word': b, 'source_sense': source, 'target_sense': target,
              'source_definition': graph['nodes'][source]['definition'],
              'target_definition': graph['nodes'][target]['definition'],
              'sense_selection': 'explicit overrides; otherwise first indexed source and same-POS target if available',
              'found': False, 'distance': None, 'path': [], 'score': None, 'contains_antonym': False,
              'interpretation': 'No declared path found within the search bounds; not evidence of no relationship.',
              'max_hops': max_hops, 'max_nodes': max_nodes, 'expanded_nodes': 0}
    if source == target:
        result.update(found=True, distance=0, score=1., interpretation='same sense' if a == b else 'same-sense synonyms')
        return result
    adjacency = defaultdict(list)
    for edge in graph['edges']:
        adjacency[edge['source']].append(edge)
    queue = deque([(source, a, [])])
    seen = {(source, a)}
    while queue and result['expanded_nodes'] < max_nodes:
        current, focused, path = queue.popleft()
        result['expanded_nodes'] += 1
        if len(path) >= max_hops:
            continue
        for edge in sorted(adjacency[current], key=lambda e: (-_WEIGHTS.get(e['relation'], .5),
                                                            e['target'], e.get('target_lemma', ''), e['relation'])):
            if 'source_lemma' in edge and focused and normalize_word(edge['source_lemma']) != focused:
                continue
            next_id = edge['target']
            next_focus = normalize_word(edge['target_lemma']) if 'target_lemma' in edge else ''
            next_path = path + [dict(edge)]
            if next_id == target and (not next_focus or next_focus == b):
                opposition = any(link['relation'] == 'antonym' for link in next_path)
                score = math.prod(_WEIGHTS.get(link['relation'], .5) for link in next_path) * .7 ** (len(next_path) - 1)
                result.update(found=True, distance=len(next_path), path=next_path, score=score,
                              contains_antonym=opposition,
                              interpretation=('opposition (antonym)' if opposition and len(next_path) == 1 else
                                              'relation path includes an antonym; not a similarity assertion' if opposition else
                                              'typed graph relatedness; heuristic score is not a probability or synonym claim'))
                return result
            key = (next_id, next_focus)
            if key not in seen:
                seen.add(key)
                queue.append((next_id, next_focus, next_path))
    return result
