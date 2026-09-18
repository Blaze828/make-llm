"""Stream a MediaWiki pages-articles dump into the prepare.py record format.

Wikitext is stripped with pattern rules, not a MediaWiki parser. Template
expansion, table content and infobox values are discarded rather than rendered,
so the output is body prose only. This is lossy on purpose; see docs/data-sources.md.
"""
import argparse
import bz2
import html
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree

# Link targets that are metadata rather than prose, in Korean and English.
DROP_PREFIX = re.compile(r"^(파일|그림|이미지|분류|틀|File|Image|Category|Template|미디어|Media):", re.I)
REDIRECT = re.compile(r"^\s*#\s*(redirect|넘겨주기)", re.I)
# Everything from these headings on is a link list, not prose.
TAIL = re.compile(r"^(각주|주석|참고 ?문헌|외부 ?링크|같이 ?보기|더 ?보기|참조|출처)\s*$", re.M)
STRIP_TAGS = ("gallery", "table", "math", "score", "timeline", "syntaxhighlight", "source", "imagemap")


def strip_braced(text, opener, closer):
    """Remove nested {{...}} or {|...|} spans that regex cannot match.

    Real dumps contain unclosed openers. Dropping everything after one would
    silently truncate the article, so an unterminated span keeps its body and
    loses only the opener token.
    """
    out, depth, i, start = [], 0, 0, 0
    while i < len(text):
        if text.startswith(opener, i):
            if not depth:
                out.append(text[start:i])
                start = i
            depth += 1
            i += len(opener)
        elif depth and text.startswith(closer, i):
            depth -= 1
            i += len(closer)
            if not depth:
                start = i
        else:
            i += 1
    out.append(text[start + len(opener):] if depth else text[start:])
    return "".join(out)


def strip_links(text):
    """Resolve [[target|label]] to label and drop file/category links entirely."""
    out, i = [], 0
    while i < len(text):
        if text.startswith("[[", i):
            depth, j = 1, i + 2
            while j < len(text) and depth:
                if text.startswith("[[", j):
                    depth += 1
                    j += 2
                elif text.startswith("]]", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth:
                # Unclosed link: treat the opener as noise and keep scanning the
                # rest, instead of emitting the remaining article unprocessed.
                i += 2
                continue
            inner = text[i + 2:j - 2]
            if not DROP_PREFIX.match(inner.strip()):
                label = inner.split("|")[-1] if "|" in inner else inner
                out.append(strip_links(label))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def clean_wikitext(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text, flags=re.I)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S | re.I)
    for tag in STRIP_TAGS:
        text = re.sub(r"<" + tag + r"[^>]*>.*?</" + tag + r">", "", text, flags=re.S | re.I)
    text = strip_braced(text, "{|", "|}")
    text = strip_braced(text, "{{", "}}")
    text = strip_links(text)
    text = re.sub(r"\[(?:https?|ftp)://\S+?\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[(?:https?|ftp)://\S+?\]", "", text)
    # Real wikitext contains unclosed external links; drop the URL, keep the label after it.
    text = re.sub(r"\[(?:https?|ftp)://\S+", "", text)
    # Tag-shaped only: a bare "a < b > c" inequality in a maths article must survive.
    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>", "", text)
    text = html.unescape(text)
    text = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.M)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^\s*-{4,}\s*$", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_apparatus(text):
    match = TAIL.search(text)
    return text[:match.start()].strip() if match else text


def page_namespace(tag):
    """The dump root declares a versioned namespace; derive it from the tag itself."""
    return tag[:tag.index("}") + 1] if "}" in tag else ""


def articles(stream):
    namespace = None
    for event, element in ElementTree.iterparse(stream, events=("end",)):
        if namespace is None and element.tag.endswith("siteinfo"):
            namespace = page_namespace(element.tag)
        if not element.tag.endswith("}page") and element.tag != "page":
            continue
        prefix = namespace if namespace is not None else page_namespace(element.tag)
        ns = element.findtext(prefix + "ns")
        title = element.findtext(prefix + "title") or ""
        page_id = element.findtext(prefix + "id")
        revision = element.find(prefix + "revision")
        body = revision.findtext(prefix + "text") if revision is not None else None
        redirect = element.find(prefix + "redirect") is not None
        element.clear()
        if ns != "0" or redirect or not body or not page_id:
            continue
        if REDIRECT.match(body):
            continue
        yield page_id, title, body


class Bunzip:
    """Incremental bz2 reader tolerant of a truncated multi-stream prefix."""

    def __init__(self, raw):
        self.raw = raw
        self.decompressor = bz2.BZ2Decompressor()
        self.buffer = b""
        self.done = False

    def read(self, size=65536):
        while not self.buffer and not self.done:
            chunk = self.raw.read(size)
            if not chunk:
                self.done = True
                break
            try:
                self.buffer += self.decompressor.decompress(chunk)
                if self.decompressor.eof:
                    leftover = self.decompressor.unused_data
                    self.decompressor = bz2.BZ2Decompressor()
                    if leftover:
                        self.buffer += self.decompressor.decompress(leftover)
            except OSError:
                # Truncated tail of a deliberately cut stream.
                self.done = True
                break
        out, self.buffer = self.buffer, b""
        return out


def convert(raw, output, source, license_id, domain, max_documents, min_chars):
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"scanned": 0, "empty_after_strip": 0, "short": 0, "kept": 0, "utf8_bytes": 0}
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        try:
            for page_id, title, body in articles(Bunzip(raw)):
                stats["scanned"] += 1
                text = drop_apparatus(clean_wikitext(body))
                if not text:
                    stats["empty_after_strip"] += 1
                    continue
                if len(text) < min_chars:
                    stats["short"] += 1
                    continue
                full = title + "\n\n" + text
                record = {"id": source + "-" + page_id, "source": source, "license": license_id,
                          "domain": domain, "text": full,
                          "group_id": source + ":" + page_id, "content_type": "text"}
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                stats["utf8_bytes"] += len(full.encode("utf-8"))
                if stats["kept"] >= max_documents:
                    break
        except ElementTree.ParseError:
            # Expected when the download is cut short on purpose.
            stats["truncated_stream"] = True
    return stats


def main():
    parser = argparse.ArgumentParser(description="MediaWiki dump to prepare.py JSONL")
    parser.add_argument("--input", required=True, help="path to .xml.bz2, or - for stdin")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True, help="dated source id, e.g. kowiki-20260901")
    parser.add_argument("--license", required=True, help="identifier from docs/data-sources.md")
    parser.add_argument("--domain", default="korean_general")
    parser.add_argument("--max-documents", type=int, default=30000)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()
    if min(args.max_documents, args.min_chars) <= 0:
        parser.error("max-documents and min-chars must be positive")
    raw = sys.stdin.buffer if args.input == "-" else open(args.input, "rb")
    try:
        stats = convert(raw, args.output, args.source, args.license, args.domain,
                        args.max_documents, args.min_chars)
    finally:
        if raw is not sys.stdin.buffer:
            raw.close()
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
