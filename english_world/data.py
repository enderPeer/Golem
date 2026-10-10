"""Small, reproducible English datasets using only the Python standard library.

Downloaded corpora and generated datasets belong in an ignored data directory.
The ``english`` preset uses the small *upstream validation* file of TinyStories
as a source corpus, then makes its own document-disjoint experimental splits.
It is therefore not the official TinyStories train/test benchmark.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?:;\"'()-\n"
CODE_ALPHABET = "".join(chr(value) for value in range(32, 127)) + "\n\t"
SELF_CODE_FILES = (
    "english_world/__init__.py", "english_world/__main__.py",
    "english_world/model.py", "english_world/evolution.py", "english_world/data.py",
    "english_world/cli.py", "ops/worker_guard.py",
)
SCHEMA_VERSION = 1
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "original_english.json"
TINY_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINY_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{TINY_REVISION}/TinyStories-valid.txt"
TINY_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
TINY_CARD_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{TINY_REVISION}/README.md"
SCOWL_URL = "https://downloads.sourceforge.net/project/wordlist/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz"
SCOWL_MIRROR = "https://deb.debian.org/debian/pool/main/s/scowl/scowl_2020.12.07.orig.tar.gz"
SCOWL_SHA256 = "5587667caa20c4891390c2d42dbb4d5c4c3f41bee77af1457ece3ba23fb859cc"
CDLA_URL = "https://raw.githubusercontent.com/spdx/license-list-data/v3.26.0/text/CDLA-Sharing-1.0.txt"
CDLA_SHA256 = "6419d7a7245cb8a9da6e3dd925d05d824ea0268e3527095f68dec091f84f75da"
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def normalize_text(text: str) -> str:
    """Use the model's small alphabet while retaining document-local newlines."""
    text = text.lower().translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\r": "\n",
    }))
    text = "".join(char if char in ALPHABET else " " for char in text)
    return "\n".join(re.sub(r" +", " ", line).strip() for line in text.splitlines()).strip()


def normalize_code(text: str) -> str:
    """Preserve source case and indentation; escape unsupported characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    return "".join(char if char in CODE_ALPHABET else f"\\x{ord(char):02x}" for char in text)


def _source_identity(records: list[dict[str, Any]]) -> str:
    identity = [{"path": record["path"], "sha256": record["sha256"]}
                for record in sorted(records, key=lambda record: record["path"])]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_source_metadata(project_root: Path) -> tuple[set[str], str | None, bool]:
    """Read only tracked-path metadata for the fixed project-source allowlist."""
    try:
        tracked = subprocess.run(["git", "-C", str(project_root), "ls-files", "-z", "--", *SELF_CODE_FILES],
                                 check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("self-code requires a Git checkout with tracked project Python files") from error
    paths = set(tracked.stdout.decode("utf-8").split("\0")) - {""}
    revision = subprocess.run(["git", "-C", str(project_root), "rev-parse", "--verify", "HEAD"],
                              capture_output=True, text=True, timeout=10)
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Unexpected Git commit identifier")
    status = subprocess.run(["git", "-C", str(project_root), "diff", "--quiet", "HEAD", "--", *SELF_CODE_FILES],
                            capture_output=True, timeout=10)
    return paths, commit, status.returncode == 0


def _safe_source_path(root: Path, relative: str) -> Path:
    if relative not in SELF_CODE_FILES:
        raise ValueError(f"Source path is outside the fixed project allowlist: {relative}")
    candidate = root / relative
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes the project or is a symbolic link: {relative}")
    return candidate


def _verify_code_snapshot(output: Path, dataset: dict[str, Any]) -> None:
    """Validate saved bytes without inspecting a changing live checkout."""
    manifest = dataset.get("manifest", {})
    records = manifest.get("source_files")
    if manifest.get("source") != "self-code" or manifest.get("domain") != "code" or not isinstance(records, list):
        raise ValueError("Code dataset is missing its source snapshot manifest")
    if not records or len({record.get("path") for record in records}) != len(records):
        raise ValueError("Invalid or duplicate snapshot source paths")
    if _source_identity(records) != manifest.get("source_sha256"):
        raise ValueError("Source snapshot manifest identity mismatch")
    texts_by_split: dict[str, set[str]] = {split: set() for split in ("train", "validation", "test")}
    snapshot = output / "source_snapshot"
    for record in records:
        relative = record["path"]
        if record.get("snapshot_path") != "source_snapshot/" + relative:
            raise ValueError("Invalid source snapshot path")
        path = _safe_source_path(snapshot, relative)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.get("sha256") or len(raw) != record.get("bytes"):
            raise ValueError(f"Source snapshot checksum mismatch: {relative}")
        text = normalize_code(raw.decode("utf-8-sig"))
        if hashlib.sha256(text.encode()).hexdigest() != record.get("normalized_sha256"):
            raise ValueError(f"Normalized source snapshot checksum mismatch: {relative}")
        split = record.get("split")
        if split not in texts_by_split:
            raise ValueError("Invalid source split")
        texts_by_split[split].add(text)
    for split, texts in texts_by_split.items():
        if not texts or texts != set(dataset.get("texts", {}).get(split, [])):
            raise ValueError(f"Snapshot text does not match the {split} split")
        if any(texts & other for name, other in texts_by_split.items() if name != split):
            raise ValueError("Identical source text appears in different splits")
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Saved manifest differs from the dataset manifest")


def _prepare_self_code(output: Path, *, seed: int, max_examples: int, context_length: int,
                       project_root: Path) -> Path:
    path = output / "dataset.json"
    if path.exists():
        saved = load_dataset(path)
        expected = {"source": "self-code", "seed": seed, "max_examples": max_examples,
                    "context_length": context_length}
        if any(saved.get("manifest", {}).get(key) != value for key, value in expected.items()):
            raise ValueError("Existing dataset has different settings; choose a new output directory for a new snapshot")
        return path
    tracked, commit, matches_commit = _git_source_metadata(project_root)
    raw_files: dict[str, bytes] = {}
    normalized: dict[str, str] = {}
    omitted = []
    for relative in SELF_CODE_FILES:
        if relative not in tracked:
            continue
        source = _safe_source_path(project_root, relative)
        if not source.is_file():
            raise ValueError(f"Tracked project source file is missing: {relative}")
        if source.stat().st_size > 4 * 1024 * 1024:
            raise ValueError(f"Project source exceeds the 4 MiB per-file limit: {relative}")
        raw = source.read_bytes()
        text = normalize_code(raw.decode("utf-8-sig"))
        if len(text) < 100:
            omitted.append({"path": relative, "reason": "source stub shorter than 100 normalized characters"})
            continue
        raw_files[relative] = raw
        normalized[relative] = text
    groups: dict[str, list[str]] = {}
    for relative, text in normalized.items():
        groups.setdefault(text, []).append(relative)
    if len(groups) < 5:
        raise ValueError("self-code requires at least five distinct tracked allowlisted source files of 100 characters")
    remaining = sorted(groups, key=lambda text: min(groups[text]))
    random.Random(seed).shuffle(remaining)
    selected = {}
    for split, preferred in (("validation", "english_world/cli.py"), ("test", "ops/worker_guard.py")):
        text = next((text for text in remaining if preferred in groups[text]), remaining[0])
        selected[split] = [text]
        remaining.remove(text)
    texts = {"train": remaining, "validation": selected["validation"], "test": selected["test"]}
    assigned = {relative: split for split, docs in texts.items() for text in docs for relative in groups[text]}
    records = [{"path": relative, "sha256": hashlib.sha256(raw_files[relative]).hexdigest(),
                "normalized_sha256": hashlib.sha256(normalized[relative].encode()).hexdigest(),
                "split": assigned[relative], "snapshot_path": "source_snapshot/" + relative,
                "bytes": len(raw_files[relative])} for relative in sorted(raw_files)]
    examples = {split: _examples(docs, max_examples if split == "train" else max(64, max_examples // 4),
                                 context_length, f"{seed}:{split}") for split, docs in texts.items()}
    manifest = {"source": "self-code", "domain": "code", "seed": seed, "max_examples": max_examples,
                "context_length": context_length, "source_commit": commit, "source_matches_commit": matches_commit,
                "source_sha256": _source_identity(records), "source_files": records, "omitted_files": omitted,
                "source_allowlist": list(SELF_CODE_FILES),
                "normalization": "UTF-8 source, BOM removed; CRLF/CR to LF; case, spaces, indentation and tabs preserved; non-ASCII/control characters escaped",
                "split_policy": "whole files; normalized exact duplicates grouped; cli.py validation and worker_guard.py test when available; remaining files train",
                "evaluation_policy": "test source files and examples are not for evolutionary selection; snapshot bytes never change during a run",
                "dictionary": {"name": "none (code domain)", "word_count": 0, "spelling_authority": False},
                "provenance": [{"name": "allowlisted tracked project Python source snapshot", "license": "project license",
                                "source_commit": commit, "source_sha256": _source_identity(records)}],
                "warnings": ["Source prediction is not evidence of executable code improvements or self-editing."]}
    manifest["splits"] = {split: {"documents": len(docs), "characters": sum(map(len, docs)),
                                  "examples": len(examples[split]),
                                  "source_paths": sorted(relative for relative, assigned_split in assigned.items() if assigned_split == split),
                                  "document_sha256": [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]}
                           for split, docs in texts.items()}
    dataset = {"schema_version": SCHEMA_VERSION, "domain": "code", "alphabet": CODE_ALPHABET,
               **examples, "dictionary": [], "texts": texts, "manifest": manifest}
    # Write precisely the bytes that were hashed above, not another read of files
    # which might have changed while preparation was running.
    output.mkdir(parents=True, exist_ok=True)
    for record in records:
        destination = _safe_source_path(output / "source_snapshot", record["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw_files[record["path"]])
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(path, dataset)
    _verify_code_snapshot(output, dataset)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_file(url: str, cache_dir: str | Path, *, expected_sha256: str | None = None,
               max_bytes: int = 64 * 1024 * 1024) -> tuple[Path, dict[str, Any]]:
    """Download a bounded HTTPS file, or verify and reuse its cached copy.

    Interrupted downloads never become valid cache entries. Checksums are
    checked on cache hits too. URLs are hashed into filenames to avoid clashes.
    """
    if not url.startswith("https://"):
        raise ValueError("Dataset downloads require an https:// URL")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    destination = cache / (key + ".download")
    metadata_path = cache / (key + ".json")
    if destination.exists():
        if destination.stat().st_size > max_bytes:
            raise ValueError(f"Cached file exceeds download limit: {destination}")
        digest = _sha256(destination)
        stored = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        recorded = expected_sha256 or stored.get("sha256")
        if not recorded or digest != recorded:
            raise ValueError(f"Cached checksum mismatch or missing checksum: {destination}")
        return destination, {"url": url, "sha256": digest, "bytes": destination.stat().st_size}
    request = urllib.request.Request(url, headers={"User-Agent": "english-world-research/0.1"})
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise ValueError(f"Download exceeds {max_bytes} byte limit: {url}")
            with tempfile.NamedTemporaryFile(mode="wb", dir=cache, suffix=".partial", delete=False) as handle:
                temporary = Path(handle.name)
                total = 0
                digest = hashlib.sha256()
                while True:
                    block = response.read(min(1024 * 1024, max_bytes - total + 1))
                    if not block:
                        break
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError(f"Download exceeds {max_bytes} byte limit: {url}")
                    handle.write(block)
                    digest.update(block)
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            raise ValueError(f"Checksum mismatch for {url}: {actual}")
        metadata = {"url": url, "sha256": actual, "bytes": total}
        temporary.replace(destination)
        _atomic_json(metadata_path, metadata)
        return destination, metadata
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _split_documents(documents: list[str], seed: int) -> dict[str, list[str]]:
    # Content deduplication precedes splitting, preventing identical documents
    # from appearing in more than one split even if upstream IDs differ.
    documents = sorted({normalize_text(doc) for doc in documents if len(normalize_text(doc)) >= 20})
    if len(documents) < 6:
        raise ValueError("At least six distinct documents of 20 characters are required")
    random.Random(seed).shuffle(documents)
    evaluation_count = max(1, len(documents) // 10)
    return {"validation": documents[:evaluation_count],
            "test": documents[evaluation_count:2 * evaluation_count],
            "train": documents[2 * evaluation_count:]}


def _examples(documents: list[str], count: int, context_length: int, seed: str) -> list[dict[str, str]]:
    ends: list[int] = []
    total = 0
    for document in documents:
        total += max(0, len(document) - 1)
        ends.append(total)
    positions = random.Random(seed).sample(range(total), min(count, total))
    result = []
    for position in positions:
        index = bisect.bisect_right(ends, position)
        offset = position - (ends[index - 1] if index else 0) + 1
        document = documents[index]
        result.append({"context": document[max(0, offset - context_length):offset],
                       "target": document[offset]})
    return result


def _read_scowl(archive: Path) -> tuple[list[str], str, list[str]]:
    words: set[str] = set()
    selected = []
    copyright_text = ""
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            if not member.isfile():
                continue
            name = member.name
            match = re.fullmatch(r"scowl-2020\.12\.07/final/(?:english|american)-(?:words|contractions)\.(\d+)", name)
            if not (match and int(match.group(1)) <= 60) and name != "scowl-2020.12.07/Copyright":
                continue
            if member.size > 4 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large SCOWL member: {name}")
            stream = package.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable SCOWL member: {name}")
            text = stream.read().decode("iso-8859-1")
            if name.endswith("/Copyright"):
                copyright_text = text
            else:
                selected.append(name)
                words.update(line.lower() for line in text.splitlines()
                             if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", line))
    if not copyright_text or len(words) < 1000:
        raise ValueError("SCOWL archive is missing its license or expected word lists")
    return sorted(words), copyright_text, sorted(selected)


def prepare_dataset(output_dir: str | Path, source: str = "fixture", *, seed: int = 1,
                    max_examples: int = 8192, context_length: int = 128,
                    cache_dir: str | Path | None = None, max_download_bytes: int = 64 * 1024 * 1024,
                    max_documents: int = 2048, allow_dictionary_fallback: bool = True,
                    project_root: str | Path | None = None) -> Path:
    """Create dataset.json and return its path; identical inputs are deterministic.

    ``max_examples`` caps training examples. Validation and test each receive
    up to max(64, max_examples // 4), with no document crossing split boundaries.
    Original fixture dialogues are included and disclosed in the English preset.
    Download or checksum failures in TinyStories are fatal. SCOWL unavailability
    may explicitly fall back to a training-only vocabulary, never held-out text.
    """
    if source not in {"fixture", "english", "self-code"}:
        raise ValueError("source must be 'fixture', 'english' or 'self-code'")
    if source == "self-code":
        if max_examples < 1 or context_length < 1:
            raise ValueError("max_examples and context_length must be positive")
        root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        return _prepare_self_code(Path(output_dir), seed=seed, max_examples=max_examples,
                                  context_length=context_length, project_root=root.resolve())
    if max_examples < 1 or context_length < 1 or max_documents < 6:
        raise ValueError("max_examples/context_length must be positive; max_documents must be >= 6")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    licenses = output / "licenses"
    licenses.mkdir(exist_ok=True)
    cache = Path(cache_dir) if cache_dir is not None else output / "cache"
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture_documents = [item["text"] for item in fixture["documents"]]
    provenance: list[dict[str, Any]] = [{"name": "original project fixtures", "path": "fixtures/original_english.json",
                                      "sha256": _sha256(FIXTURE), "license": "project license",
                                      "documents": len(fixture_documents)}]
    manifest: dict[str, Any] = {"source": source, "seed": seed, "context_length": context_length,
                                "max_examples": max_examples, "provenance": provenance, "warnings": [],
                                "normalization": "lowercase restricted ASCII; Unicode quotes/dashes normalized",
                                "split_policy": "whole normalized documents; exact duplicates removed before seeded 80/10/10 split",
                                "evaluation_policy": "test examples are not for evolutionary selection"}
    documents = list(fixture_documents)
    if source == "english":
        corpus, info = fetch_file(TINY_URL, cache, expected_sha256=TINY_SHA256, max_bytes=max_download_bytes)
        stories = [normalize_text(text) for text in corpus.read_text(encoding="utf-8").split("<|endoftext|>")]
        stories = sorted({text for text in stories if len(text) >= 20})
        random.Random(seed).shuffle(stories)
        selected_stories = stories[:max_documents]
        documents.extend(selected_stories)
        info.update({"name": "TinyStories", "revision": TINY_REVISION, "upstream_file": "TinyStories-valid.txt",
                     "license": "CDLA-Sharing-1.0", "license_url": "https://cdla.dev/sharing-1-0/",
                     "available_documents": len(stories), "selected_documents": len(selected_stories),
                     "modifications": "selected subset; lowercase/restricted alphabet; local document splits; mixed with original fixtures"})
        provenance.append(info)
        manifest["warnings"].append("Local experimental splits use the upstream validation corpus; these are not official TinyStories benchmark scores.")
        license_file, license_info = fetch_file(CDLA_URL, cache, expected_sha256=CDLA_SHA256, max_bytes=max_download_bytes)
        shutil.copyfile(license_file, licenses / "TinyStories-CDLA-Sharing-1.0.txt")
        card, card_info = fetch_file(TINY_CARD_URL, cache, max_bytes=max_download_bytes)
        shutil.copyfile(card, licenses / "TinyStories-upstream-README.md")
        info["license_copy"] = "licenses/TinyStories-CDLA-Sharing-1.0.txt"
        info["license_sha256"] = license_info["sha256"]
        info["dataset_card_sha256"] = card_info["sha256"]
    texts = _split_documents(documents, seed)
    dictionary = sorted(set(WORD_RE.findall("\n".join(texts["train"]))))
    manifest["dictionary"] = {"name": "training vocabulary", "spelling_authority": False,
                               "note": "Observed words from training documents only; not an independent spelling dictionary."}
    if source == "english":
        try:
            try:
                archive, info = fetch_file(SCOWL_URL, cache, expected_sha256=SCOWL_SHA256, max_bytes=max_download_bytes)
            except (urllib.error.URLError, TimeoutError, OSError):
                archive, info = fetch_file(SCOWL_MIRROR, cache, expected_sha256=SCOWL_SHA256, max_bytes=max_download_bytes)
            dictionary, copyright_text, members = _read_scowl(archive)
            (licenses / "SCOWL-Copyright.txt").write_text(copyright_text, encoding="utf-8")
            info.update({"name": "SCOWL", "version": "2020.12.07", "license": "SCOWL notices (permissive composite)",
                         "license_copy": "licenses/SCOWL-Copyright.txt", "selected_members": members})
            provenance.append(info)
            manifest["dictionary"] = {"name": "SCOWL 2020.12.07 English/American words and contractions, sizes <= 60",
                                       "spelling_authority": True,
                                       "note": "Dictionary membership is only a spelling proxy; names, dialects and missing entries may be rejected."}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, tarfile.TarError) as error:
            if not allow_dictionary_fallback:
                raise
            manifest["warnings"].append(f"Independent SCOWL dictionary unavailable; using training vocabulary only: {error}")
    examples = {split: _examples(docs, max_examples if split == "train" else max(64, max_examples // 4),
                                 context_length, f"{seed}:{split}") for split, docs in texts.items()}
    manifest["splits"] = {split: {"documents": len(docs), "characters": sum(map(len, docs)),
                                  "examples": len(examples[split]),
                                  "document_sha256": [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]}
                           for split, docs in texts.items()}
    manifest["dictionary"]["word_count"] = len(dictionary)
    result = {"schema_version": SCHEMA_VERSION, "alphabet": ALPHABET, **examples,
              "dictionary": dictionary, "texts": texts, "manifest": manifest}
    path = output / "dataset.json"
    _atomic_json(path, result)
    _atomic_json(output / "manifest.json", manifest)
    return path


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load and validate a prepared dataset or a directory containing one."""
    path = Path(path)
    if path.is_dir():
        path = path / "dataset.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema version")
    domain = result.get("domain", "english")
    if domain not in {"english", "code"}:
        raise ValueError("Unsupported dataset domain")
    alphabet = CODE_ALPHABET if domain == "code" else ALPHABET
    if result.get("alphabet") != alphabet:
        raise ValueError("Dataset alphabet differs from its domain alphabet")
    for split in ("train", "validation", "test"):
        if not result.get(split):
            raise ValueError(f"Missing or empty {split} examples")
        for example in result[split]:
            if not isinstance(example, dict) or not isinstance(example.get("context"), str):
                raise ValueError(f"Invalid {split} example")
            target = example.get("target")
            if not isinstance(target, str) or len(target) != 1 or target not in alphabet:
                raise ValueError(f"Invalid target in {split}")
            if not example["context"] or any(char not in alphabet for char in example["context"]):
                raise ValueError(f"Invalid context in {split}")
    if not isinstance(result.get("dictionary"), list) or (domain == "english" and not result["dictionary"]):
        raise ValueError("Missing dictionary")
    if domain == "code":
        if result["dictionary"]:
            raise ValueError("Code datasets must not contain a spelling dictionary")
        _verify_code_snapshot(path.parent, result)
    if any(not isinstance(word, str) or WORD_RE.fullmatch(word) is None for word in result["dictionary"]):
        raise ValueError("Invalid dictionary entry")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("fixture", "english", "self-code"), default="fixture")
    parser.add_argument("--output", default="data/fixture")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-documents", type=int, default=2048)
    parser.add_argument("--require-dictionary", action="store_true")
    args = parser.parse_args()
    path = prepare_dataset(args.output, args.source, seed=args.seed, max_examples=args.max_examples,
                           context_length=args.context_length, max_documents=args.max_documents,
                           allow_dictionary_fallback=not args.require_dictionary)
    dataset = load_dataset(path)
    print(json.dumps({"dataset": str(path.resolve()), "examples": {s: len(dataset[s]) for s in ("train", "validation", "test")},
                      "dictionary": dataset["manifest"]["dictionary"], "warnings": dataset["manifest"]["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
