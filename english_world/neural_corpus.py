"""Pinned, small OpenAssistant dialogue corpus with conversation-group splits.

Only reviewed English ready-for-export messages are eligible. Conversation trees
(including duplicated root prompts) are assigned to splits BEFORE a single
best-ranked branch is extracted. Original archive, license and card identities
are recorded; downloaded data remain outside the public source repository.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil

from .data import _atomic_json, _sha256, fetch_file

SCHEMA = 'golem-neural-corpus-v1'
SOURCE_REVISION = 'fdf72ae0827c1cda404aff25b6603abec9e3399b'
SOURCE_FILE = '2023-04-12_oasst_ready.messages.jsonl.gz'
SOURCE_SHA256 = '286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c'
SOURCE_BASE = f'https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/{SOURCE_REVISION}'
MAX_DOWNLOAD = 64 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
SPLITS = ('train', 'validation', 'test')


def _text(value: str) -> str:
    return value.replace('\r\n', '\n').replace('\r', '\n').strip('\n')


def _label(message: dict, name: str, default: float = 0.) -> float:
    labels = message.get('labels') or {}
    if isinstance(labels, dict) and isinstance(labels.get(name), dict):
        value = labels[name].get('value', default)
    elif isinstance(labels, dict) and isinstance(labels.get('name'), list):
        mapping = dict(zip(labels['name'], labels.get('value', [])))
        value = mapping.get(name, default)
    else:
        value = default
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def _eligible(message: dict) -> bool:
    return (isinstance(message.get('message_id'), str) and isinstance(message.get('message_tree_id'), str)
            and message.get('lang') == 'en' and message.get('tree_state') == 'ready_for_export'
            and message.get('review_result') is True and message.get('deleted') is not True
            and message.get('synthetic') is not True and message.get('role') in ('prompter', 'assistant')
            and isinstance(message.get('text'), str) and bool(message['text'].strip())
            and len(message['text'].encode('utf-8')) <= 32768
            and _label(message, 'quality') >= .5
            and all(_label(message, name) < .5 for name in ('spam', 'pii', 'lang_mismatch', 'fails_task')))


def _rank(message: dict):
    rank = message.get('rank')
    return (rank if type(rank) is int and rank >= 0 else 1000000,
            -_label(message, 'quality'), -_label(message, 'helpfulness'), message['message_id'])


def _read_messages(archive: Path) -> list[dict]:
    messages, size = [], 0
    with gzip.open(archive, 'rb') as source:
        for line in source:
            size += len(line)
            if size > MAX_EXPANDED or len(line) > 1024 * 1024:
                raise ValueError('expanded OpenAssistant archive exceeds its size limit')
            message = json.loads(line.decode('utf-8'))
            if _eligible(message):
                messages.append(message)
    return messages


def _root_key(message: dict) -> str:
    # Group even case/whitespace variants before splitting; source text itself is
    # preserved. This is more conservative than exact rendered-branch deduping.
    return hashlib.sha256(' '.join(_text(message['text']).split()).casefold().encode('utf-8')).hexdigest()


def _render(turns: list[dict]) -> str:
    rendered = []
    for turn in turns:
        # Internal quoted role-looking lines are content, not new message roles.
        content = re.sub(r'(?im)^(user|assistant):', r' \1:', turn['content'])
        rendered.append(turn['role'] + ': ' + content)
    return '\n'.join(rendered)


def _build_conversations(messages: list[dict], seed: int, max_conversations: int) -> tuple[dict, list[dict], dict]:
    by_id = {}
    for message in messages:
        if not _eligible(message):
            continue
        identity = message['message_id']
        if identity in by_id and by_id[identity] != message:
            raise ValueError('conflicting duplicate OpenAssistant message ID')
        by_id[identity] = message
    children = defaultdict(list)
    for message in by_id.values():
        parent = by_id.get(message.get('parent_id'))
        if parent and parent['message_tree_id'] == message['message_tree_id']:
            children[parent['message_id']].append(message)
    grouped = defaultdict(list)
    for message in by_id.values():
        if (not message.get('parent_id') and message['role'] == 'prompter'
                and any(child['role'] == 'assistant' for child in children[message['message_id']])):
            grouped[_root_key(message)].append(message)
    group_keys = sorted(grouped)
    random.Random(seed).shuffle(group_keys)
    group_keys = group_keys[:max_conversations]
    if len(group_keys) < 5:
        raise ValueError('need at least five distinct reviewed English conversation roots')
    evaluation_count = max(1, len(group_keys) // 10)
    # Assign all root/tree groups BEFORE selecting branches. Only one canonical
    # root per duplicate group is emitted, so branches cannot cross split borders.
    assignments = {key: ('validation' if i < evaluation_count else
                         'test' if i < 2 * evaluation_count else 'train')
                   for i, key in enumerate(group_keys)}
    conversations = {split: [] for split in SPLITS}
    selected_messages = {}
    for key in group_keys:
        root = min(grouped[key], key=lambda row: (-_label(row, 'quality'), row['message_id']))
        path, seen = [root], {root['message_id']}
        while len(path) < 16:
            current = path[-1]
            expected = 'assistant' if current['role'] == 'prompter' else 'prompter'
            options = [child for child in children[current['message_id']]
                       if child['role'] == expected and child['message_id'] not in seen]
            if expected == 'prompter':
                options = [child for child in options if any(reply['role'] == 'assistant'
                           for reply in children[child['message_id']])]
            if not options:
                break
            winner = min(options, key=_rank)
            if sum(len(row['text'].encode('utf-8')) for row in path) + len(winner['text'].encode('utf-8')) > 128 * 1024:
                break
            path.append(winner)
            seen.add(winner['message_id'])
        if path[-1]['role'] == 'prompter':
            path.pop()
        if len(path) < 2:
            continue
        turns = [{'role': 'user' if row['role'] == 'prompter' else 'assistant', 'content': _text(row['text'])}
                 for row in path]
        record = {'tree_id': root['message_tree_id'], 'root_group_sha256': key,
                  'message_ids': [row['message_id'] for row in path], 'turns': turns}
        conversations[assignments[key]].append(record)
        for row in path:
            selected_messages[row['message_id']] = {field: row.get(field) for field in
                ('message_id', 'parent_id', 'message_tree_id', 'role', 'text', 'lang', 'rank',
                 'review_result', 'deleted', 'synthetic', 'tree_state', 'labels')}
    if any(not conversations[split] for split in SPLITS):
        raise ValueError('conversation filtering left an empty split')
    stats = {'eligible_messages': len(by_id), 'eligible_root_groups': len(grouped),
             'selected_root_groups': len(group_keys), 'selected_messages': len(selected_messages)}
    return conversations, [selected_messages[key] for key in sorted(selected_messages)], stats


def _content_hash(dataset: dict) -> str:
    content = {key: value for key, value in dataset.items() if key != 'manifest'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def prepare_neural_corpus(output_dir: str | Path, seed: int = 23, max_conversations: int = 512,
                          cache_dir: str | Path | None = None, story_path: str | Path | None = None) -> Path:
    if type(max_conversations) is not int or not 5 <= max_conversations <= 10000:
        raise ValueError('max_conversations must be an integer in 5..10000')
    output = Path(output_dir)
    dataset_path = output / 'dataset.json'
    parameters = {'seed': int(seed), 'max_conversations': max_conversations,
                  'story_dataset_sha256': _sha256(Path(story_path)) if story_path else None}
    if dataset_path.exists():
        existing = load_neural_corpus(dataset_path)
        if existing['manifest']['parameters'] != parameters:
            raise ValueError('existing corpus has different settings; choose another output directory')
        return dataset_path
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory is occupied but contains no verified neural corpus')
    cache = Path(cache_dir) if cache_dir else output.parent / '.neural-corpus-cache'
    archive, archive_meta = fetch_file(SOURCE_BASE + '/' + SOURCE_FILE, cache,
                                      expected_sha256=SOURCE_SHA256, max_bytes=MAX_DOWNLOAD)
    license_path, license_meta = fetch_file(SOURCE_BASE + '/LICENSE', cache, max_bytes=128 * 1024)
    card_path, card_meta = fetch_file(SOURCE_BASE + '/README.md', cache, max_bytes=1024 * 1024)
    conversations, selected_messages, stats = _build_conversations(_read_messages(archive), seed, max_conversations)
    texts = {split: [_render(record['turns']) for record in conversations[split]] for split in SPLITS}
    story_provenance = None
    if story_path:
        from .data import load_dataset
        stories = load_dataset(story_path)
        story_provenance = stories.get('manifest', {})
        for split in SPLITS:
            options = list(stories.get('texts', {}).get(split, []))
            random.Random(f'{seed}:{split}:stories').shuffle(options)
            texts[split].extend(options[:max(1, max_conversations // (4 if split == 'train' else 20))])
    # Exact rendered document duplicates are forbidden across all source mixtures.
    owners = {}
    for split in SPLITS:
        for text in texts[split]:
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if digest in owners and owners[digest] != split:
                raise ValueError('duplicate document across corpus splits')
            owners[digest] = split
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    selected_path = snapshot / 'selected_messages.jsonl'
    with selected_path.open('w', encoding='utf-8', newline='\n') as handle:
        for message in selected_messages:
            handle.write(json.dumps(message, ensure_ascii=False, sort_keys=True) + '\n')
    shutil.copyfile(license_path, output / 'OASST1-LICENSE.txt')
    shutil.copyfile(card_path, output / 'OASST1-DATASET-CARD.md')
    manifest = {'source': 'OpenAssistant/oasst1', 'revision': SOURCE_REVISION, 'license': 'Apache-2.0',
                'parameters': parameters, 'archive': archive_meta, 'license_file': license_meta, 'dataset_card': card_meta,
                'source_snapshot': {'path': 'source_snapshot/selected_messages.jsonl', 'sha256': _sha256(selected_path)},
                'filter': 'English, ready_for_export, review_result=true, not deleted/synthetic; quality>=0.5; '
                          'spam/pii/language-mismatch/fails-task labels<0.5; message UTF-8 <=32768 bytes',
                'selection': 'seeded root-group subset; one branch per root; lowest assistant rank, then quality/helpfulness',
                'split_policy': '80/10/10 conversation-root groups, duplicate root prompts grouped before branches extracted',
                'text_transforms': 'CRLF/CR to LF, trim outer newlines; preserve UTF-8/case/indent; indent internal role-looking lines',
                'evaluation_policy': 'validation for model selection; test conversations never used for training or mutation',
                'benchmark_note': 'Local small experimental splits, not official OASST train/validation benchmark results',
                'stats': stats, 'story_provenance': story_provenance,
                'splits': {split: {'conversations': len(conversations[split]), 'documents': len(texts[split]),
                                  'characters': sum(map(len, texts[split])),
                                  'tree_ids': [row['tree_id'] for row in conversations[split]]} for split in SPLITS}}
    dataset = {'schema': SCHEMA, 'domain': 'dialogue_utf8', 'texts': texts,
               'conversations': conversations, 'dictionary': [], 'manifest': manifest}
    manifest['content_sha256'] = _content_hash(dataset)
    _atomic_json(output / 'manifest.json', manifest)
    _atomic_json(dataset_path, dataset)
    load_neural_corpus(dataset_path)
    return dataset_path


def load_neural_corpus(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / 'dataset.json'
    dataset = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(dataset, dict) or dataset.get('schema') != SCHEMA:
        raise ValueError('unsupported neural corpus schema')
    manifest = dataset.get('manifest', {})
    if manifest.get('content_sha256') != _content_hash(dataset):
        raise ValueError('neural corpus content checksum mismatch')
    if json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8')) != manifest:
        raise ValueError('neural corpus manifest mismatch')
    snapshot = manifest.get('source_snapshot', {})
    if snapshot.get('path') != 'source_snapshot/selected_messages.jsonl':
        raise ValueError('unexpected corpus snapshot path')
    if _sha256(path.parent / snapshot['path']) != snapshot.get('sha256'):
        raise ValueError('neural source snapshot checksum mismatch')
    for filename, metadata in (('OASST1-LICENSE.txt', 'license_file'), ('OASST1-DATASET-CARD.md', 'dataset_card')):
        if _sha256(path.parent / filename) != manifest.get(metadata, {}).get('sha256'):
            raise ValueError('corpus license/card checksum mismatch')
    seen_groups, seen_trees, seen_texts = {}, {}, {}
    for split in SPLITS:
        documents = dataset.get('texts', {}).get(split)
        records = dataset.get('conversations', {}).get(split)
        if not isinstance(documents, list) or not documents or not isinstance(records, list) or not records:
            raise ValueError('neural corpus has an empty or invalid split')
        for record in records:
            for seen, key in ((seen_groups, record['root_group_sha256']), (seen_trees, record['tree_id'])):
                if key in seen and seen[key] != split:
                    raise ValueError('conversation group crosses corpus splits')
                seen[key] = split
            if _render(record['turns']) not in documents:
                raise ValueError('conversation turns do not match rendered text')
        for document in documents:
            if not isinstance(document, str) or not document:
                raise ValueError('invalid corpus document')
            digest = hashlib.sha256(document.encode('utf-8')).hexdigest()
            if digest in seen_texts and seen_texts[digest] != split:
                raise ValueError('document crosses corpus splits')
            seen_texts[digest] = split
    return dataset
