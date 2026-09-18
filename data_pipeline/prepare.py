"""Disk-backed document cleaning, deduplication and grouped split assignment."""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from .common import read_jsonl, sha256_file, write_json


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in ("p", "div", "br", "li") and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden-1)
        if tag in ("p", "div", "li") and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(record, config):
    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if record.get("content_type", "text") == "html":
        if record.get("domain") == "code":
            raise ValueError("Code records cannot use the HTML stripping path")
        parser = HTMLText(); parser.feed(text); parser.close()
        text = "".join(parser.parts)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if config["redact_email_phone"]:
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
        text = re.sub(r"(?<!\d)01[016789][- .]?\d{3,4}[- .]?\d{4}(?!\d)", "[PHONE]", text)
    return text


def canonical(text):
    # Dedup key only. Model text retains case and whitespace.
    return " ".join(unicodedata.normalize("NFC", text).split())


def shingles(text):
    return {text[i:i+5] for i in range(max(1, len(text)-4))}


def signatures(parts):
    # Eight independent minhash bands; candidates are verified with exact Jaccard.
    return [min(hashlib.blake2b((str(seed)+part).encode(), digest_size=8).hexdigest()
                for part in parts) for seed in range(8)]


def split_for(group, seed, validation_ratio, test_ratio):
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16)/2**64
    return "test" if bucket < test_ratio else "validation" if bucket < test_ratio+validation_ratio else "train"


def prepare(input_paths, output, config):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    if not config["allowed_licenses"]:
        raise ValueError("Set allowed_licenses after inspecting the actual data rights")
    if not 1 <= config["min_chars"] <= config["max_chars"]:
        raise ValueError("Invalid document size limits")
    if not 0 < config["near_duplicate_jaccard"] <= 1:
        raise ValueError("Jaccard threshold must be in (0,1]")
    vr, tr = config["validation_ratio"], config["test_ratio"]
    if not (0 < vr < 1 and 0 < tr < 1 and vr+tr < 1):
        raise ValueError("Invalid split ratios")
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output/"dedup.sqlite")
    db.executescript("""
      CREATE TABLE docs (id INTEGER PRIMARY KEY, digest TEXT UNIQUE, canonical TEXT, group_id TEXT, payload TEXT);
      CREATE INDEX groups_idx ON docs(group_id);
      CREATE TABLE bands (band INTEGER, signature TEXT, doc_id INTEGER);
      CREATE INDEX bands_idx ON bands(band,signature);
      CREATE TABLE aliases (external_group TEXT PRIMARY KEY, group_id TEXT);
    """)
    stats = Counter()
    try:
        with (output/"rejected.jsonl").open("w", encoding="utf-8") as rejected:
            for path in input_paths:
                for record in read_jsonl(path):
                    stats["input"] += 1
                    reason = None
                    if not all(isinstance(record.get(k), str) and record[k] for k in ("id", "source", "license", "domain")):
                        reason = "missing_provenance"
                    elif record["license"] not in config["allowed_licenses"]:
                        reason = "license_not_allowed"
                    text = clean_text(record, config) if reason is None else ""
                    normalized = canonical(text)
                    if reason is None:
                        if not normalized or len(text) < config["min_chars"]: reason = "too_short"
                        elif len(text) > config["max_chars"]: reason = "too_long"
                        elif "\ufffd" in text or "\x00" in text: reason = "broken_text"
                        elif re.search(r"(.)\1{30,}", text): reason = "repeated_character"
                    if reason:
                        stats[reason] += 1
                        rejected.write(json.dumps({"id": record.get("id"), "reason": reason}, ensure_ascii=False)+"\n")
                        continue
                    digest = hashlib.sha256(normalized.encode()).hexdigest()
                    groups = set()
                    existing = db.execute("SELECT group_id FROM docs WHERE digest=?", (digest,)).fetchone()
                    if existing: groups.add(existing[0])
                    external = record.get("group_id")
                    if external is not None and (not isinstance(external, str) or not external):
                        raise ValueError("group_id must be a nonempty string when supplied")
                    if external:
                        found = db.execute("SELECT group_id FROM aliases WHERE external_group=?", (external,)).fetchone()
                        if found: groups.add(found[0])
                    parts, bands = None, []
                    if config["near_duplicate_grouping"] and not existing:
                        parts = shingles(normalized); bands = signatures(parts)
                        candidates = set()
                        for i, sig in enumerate(bands):
                            candidates.update(row[0] for row in db.execute("SELECT doc_id FROM bands WHERE band=? AND signature=?", (i,sig)))
                        for doc_id in candidates:
                            other, group = db.execute("SELECT canonical,group_id FROM docs WHERE id=?", (doc_id,)).fetchone()
                            other_parts = shingles(other)
                            score = len(parts & other_parts)/len(parts | other_parts)
                            if score >= config["near_duplicate_jaccard"]:
                                groups.add(group); stats["near_duplicate_links"] += 1
                    root = min(groups | {digest})
                    for group in groups:
                        if group != root:
                            db.execute("UPDATE docs SET group_id=? WHERE group_id=?", (root, group))
                            db.execute("UPDATE aliases SET group_id=? WHERE group_id=?", (root, group))
                    if external:
                        db.execute("INSERT OR REPLACE INTO aliases VALUES (?,?)", (external, root))
                    if existing:
                        stats["exact_duplicates_removed"] += 1
                        continue
                    payload = {**record, "text": text, "content_sha256": digest}
                    cursor = db.execute("INSERT INTO docs(digest,canonical,group_id,payload) VALUES (?,?,?,?)",
                                        (digest, normalized, root, json.dumps(payload, ensure_ascii=False)))
                    db.executemany("INSERT INTO bands VALUES (?,?,?)", [(i,sig,cursor.lastrowid) for i,sig in enumerate(bands)])
                    if stats["input"] % 1000 == 0: db.commit()
        db.commit()
        files = {name: (output/f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") for name in ("train","validation","test")}
        sources = Counter()
        try:
            for group, payload in db.execute("SELECT group_id,payload FROM docs ORDER BY digest"):
                row = json.loads(payload)
                split = split_for(group, config["seed"], vr, tr)
                row.update(group_id=group, split=split)
                files[split].write(json.dumps(row, ensure_ascii=False)+"\n")
                stats[f"{split}_documents"] += 1
                stats[f"{split}_utf8_bytes"] += len(row["text"].encode())
                sources[f"{row['source']}:{row['domain']}"] += 1
        finally:
            for stream in files.values(): stream.close()
        manifest = {"format_version":1, "kind":"clean_corpus", "config":config,
                    "inputs":[{"name":Path(p).name,"sha256":sha256_file(p)} for p in input_paths],
                    "stats":dict(stats), "sources":dict(sources),
                    "splits":{name:{"path":f"{name}.jsonl","sha256":sha256_file(output/f"{name}.jsonl")} for name in files},
                    "limitations":["minhash candidate search can miss near duplicates", "PII patterns only cover email and Korean mobile phone numbers"]}
        write_json(output/"manifest.json", manifest)
        return manifest
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/data/preparation.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(prepare(args.input,args.output,config)["stats"], ensure_ascii=False))


if __name__ == "__main__": main()
