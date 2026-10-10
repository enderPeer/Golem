"""Portable graph exports and a small, explicitly sense-specific neighborhood view."""
from __future__ import annotations

import csv
import html
from pathlib import Path
import textwrap

from .lexical_graph import normalize_word, rank_related


def _license_sidecar(path, license_text):
    if license_text:
        (Path(path).parent / "WORDNET-LICENSE.txt").write_text(license_text, encoding="utf-8")


def ranked_words(graph, limit=None):
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("Ranking limit must be a nonnegative integer or None")
    rows = sorted(graph["rank"]["words"].items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        rows = rows[:limit]
    return [{"rank": i + 1, "word": word, "connectivity": score,
             "meanings": len(graph["word_index"][word])} for i, (word, score) in enumerate(rows)]


def export_rank_table(graph, output, license_text=""):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("rank", "word", "connectivity", "meanings"), delimiter="\t")
        writer.writeheader()
        writer.writerows(ranked_words(graph))
    _license_sidecar(path, license_text)
    return path


def export_graphml(graph, output, license_text=""):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    escape = lambda value: html.escape(str(value), quote=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        for key, target, name, kind in (("label", "node", "lemmas", "string"), ("pos", "node", "part_of_speech", "string"),
                                        ("definition", "node", "definition", "string"), ("rank", "node", "connectivity", "double"),
                                        ("relation", "edge", "relation", "string"), ("source_lemma", "edge", "source_lemma", "string"),
                                        ("target_lemma", "edge", "target_lemma", "string"), ("license", "graph", "license", "string")):
            stream.write(f'<key id="{key}" for="{target}" attr.name="{name}" attr.type="{kind}"/>\n')
        stream.write('<graph id="wordnet" edgedefault="directed">\n')
        stream.write(f'<data key="license">{escape(license_text)}</data>\n')
        for identity, node in sorted(graph["nodes"].items()):
            stream.write(f'<node id="{escape(identity)}">')
            for key, value in (("label", ", ".join(node["lemmas"])), ("pos", node["pos"]),
                               ("definition", node["definition"]), ("rank", graph["rank"]["synsets"].get(identity, 0.))):
                stream.write(f'<data key="{key}">{escape(value)}</data>')
            stream.write('</node>\n')
        for index, edge in enumerate(graph["edges"]):
            stream.write(f'<edge id="e{index}" source="{escape(edge["source"])}" target="{escape(edge["target"])}">')
            for key in ("relation", "source_lemma", "target_lemma"):
                if key in edge:
                    stream.write(f'<data key="{key}">{escape(edge[key])}</data>')
            stream.write('</edge>\n')
        stream.write('</graph></graphml>\n')
    _license_sidecar(path, license_text)
    return path


def _neighborhood(graph, word, sense, limit):
    senses = graph["word_index"].get(normalize_word(word), [])
    if not senses:
        raise ValueError(f"WordNet has no exact lemma entry for {word!r}")
    sense = sense or senses[0]
    if sense not in senses:
        raise ValueError("Sense does not belong to requested word")
    return sense, rank_related(graph, word, sense=sense, limit=limit, max_depth=1)


def export_neighborhood(graph, word, output, sense=None, limit=10, license_text=""):
    """SVG is dependency-free; PNG is an optional local Matplotlib rendering."""
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Neighborhood limit must be 1..20")
    sense, rows = _neighborhood(graph, word, sense, limit)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        result = _png(graph, word, sense, rows, path, license_text)
        _license_sidecar(path, license_text)
        return result
    if path.suffix.lower() != ".svg":
        raise ValueError("Neighborhood output must be .svg or .png")
    height = max(500, 150 + 78 * len(rows))
    center = height / 2
    esc = lambda value: html.escape(str(value), quote=True)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}">',
             f'<title>WordNet relations for {esc(word)}, sense {esc(sense)}</title>',
             '<desc>Each arrow is a declared relation or same-sense synonym. Scores are heuristic relatedness, not probabilities.</desc>',
             f'<metadata>{esc(license_text)}</metadata>', '<rect width="100%" height="100%" fill="#fbfcfe"/>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>',
             '<g font-family="Segoe UI,Arial,sans-serif" fill="#162438">',
             f'<text x="35" y="40" font-size="25" font-weight="600">Golem · relations of {esc(word)}</text>',
             '<text x="35" y="67" font-size="14" fill="#526173">One meaning at a time · direct dictionary relations · ranked connections</text>']
    parts.append(f'<rect x="35" y="{center-100}" width="290" height="200" rx="12" fill="#e7eef8"/>')
    parts.append(f'<text x="55" y="{center-66}" font-size="23" font-weight="600">{esc(word)}</text>')
    parts.append(f'<text x="55" y="{center-40}" font-size="12">{esc(sense)} · {esc(graph["nodes"][sense]["pos"])}</text>')
    definition = graph["nodes"][sense]["definition"].split('; "')[0]
    for index, line in enumerate(textwrap.wrap(definition, 37)[:5]):
        parts.append(f'<text x="55" y="{center-13+19*index}" font-size="14">{esc(line)}</text>')
    for index, row in enumerate(rows):
        y = 120 + index * 78
        parts.append(f'<path d="M325,{center} C425,{center} 435,{y} 600,{y}" fill="none" stroke="#8b9bae" stroke-width="1.4" marker-end="url(#arrow)"/>')
        relation = row["relation"].replace('_', ' ')
        parts.append(f'<rect x="370" y="{y-12}" width="220" height="23" rx="4" fill="#fbfcfe"/>')
        parts.append(f'<text x="382" y="{y+4}" font-size="13">{esc(relation)} · {row["score"]:.2f}</text>')
        parts.append(f'<rect x="615" y="{y-28}" width="465" height="58" rx="8" fill="#edf2f7"/>')
        label = ", ".join(row["lemmas"][:3])
        if len(label) > 58:
            label = label[:55] + "..."
        parts.append(f'<text x="633" y="{y-4}" font-size="16">{esc(label)}</text>')
        parts.append(f'<text x="633" y="{y+16}" font-size="11" fill="#526173">{esc(row["target_sense"])} · sense preserved</text>')
    parts.append(f'<text x="35" y="{height-28}" font-size="12" fill="#526173">WordNet 3.0 © 2006 Princeton University · heuristic scores; centrality is not frequency or importance</text>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts), encoding="utf-8")
    _license_sidecar(path, license_text)
    return path


def _png(graph, word, sense, rows, path, license_text=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except ImportError as exc:
        raise ValueError("PNG export needs optional Matplotlib; use SVG for dependency-free export") from exc
    height = max(500, 150 + 78 * len(rows))
    fig, ax = plt.subplots(figsize=(11.2, height / 100), dpi=140)
    fig.patch.set_facecolor('#fbfcfe')
    ax.set(xlim=(0, 1120), ylim=(height, 0))
    ax.axis('off')
    ax.text(35, 36, f"Golem · relations of {word}", fontsize=18, weight='bold', color='#162438')
    ax.text(35, 66, "One meaning at a time · direct dictionary relations · ranked connections", fontsize=9, color='#526173')
    center = height / 2
    ax.add_patch(FancyBboxPatch((35, center - 100), 290, 200, boxstyle='round,pad=0,rounding_size=10', facecolor='#e7eef8', edgecolor='none'))
    ax.text(55, center - 65, word, fontsize=18, weight='bold', color='#162438')
    ax.text(55, center - 38, f"{sense} · {graph['nodes'][sense]['pos']}", fontsize=9, color='#526173')
    definition = graph['nodes'][sense]['definition'].split('; "')[0]
    ax.text(55, center - 14, '\n'.join(textwrap.wrap(definition, 35)[:5]), fontsize=10, va='top', color='#162438', linespacing=1.5)
    for index, row in enumerate(rows):
        y = 120 + index * 78
        ax.add_patch(FancyArrowPatch((325, center), (606, y), connectionstyle='arc3,rad=0', arrowstyle='-|>', mutation_scale=10, lw=1, color='#8b9bae'))
        label = row['relation'].replace('_', ' ') + f" · {row['score']:.2f}"
        ax.text(379, y, label, fontsize=9, va='center', color='#162438', bbox={'facecolor':'#fbfcfe','edgecolor':'none','pad':2})
        ax.add_patch(FancyBboxPatch((615, y - 28), 465, 58, boxstyle='round,pad=0,rounding_size=6', facecolor='#edf2f7', edgecolor='none'))
        title = ', '.join(row['lemmas'][:3])
        ax.text(633, y - 4, title if len(title) <= 58 else title[:55]+'...', fontsize=11, color='#162438')
        ax.text(633, y + 17, row['target_sense']+' · sense preserved', fontsize=8, color='#526173')
    ax.text(35, height - 22, "WordNet 3.0 © 2006 Princeton University · heuristic scores; centrality is not frequency or importance", fontsize=8, color='#526173')
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(),
                metadata={"Title": f"WordNet relations for {word}, sense {sense}",
                          "Description": "Sense-specific declared relations; scores are heuristics, not probabilities.",
                          "Copyright": license_text})
    plt.close(fig)
    return path
