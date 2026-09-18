import io
import json

import pytest

from data_pipeline.wiki_dump import clean_wikitext, convert, strip_braced, strip_links

DUMP = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11">
  <siteinfo><sitename>Wikipedia</sitename></siteinfo>
  <page><title>본문 문서</title><ns>0</ns><id>11</id><revision><id>1</id>
    <text xml:space="preserve">{{정보상자|버림}}'''본문'''은 [[링크|표시되는 말]]과 &lt;ref&gt;각주&lt;/ref&gt; 를 담는다.

== 각주 ==
* 지워질 목록</text></revision></page>
  <page><title>넘겨주기</title><ns>0</ns><id>12</id><redirect title="본문 문서" />
    <revision><id>2</id><text xml:space="preserve">#넘겨주기 [[본문 문서]]</text></revision></page>
  <page><title>토론 문서</title><ns>1</ns><id>13</id><revision><id>3</id>
    <text xml:space="preserve">다른 이름공간이므로 제외되는 충분히 긴 본문이다.</text></revision></page>
</mediawiki>
"""


def test_unclosed_markup_keeps_following_prose():
    """A malformed opener must not swallow or leak the rest of the article."""
    assert clean_wikitext("앞 [[깨진 뒤에 [[정상|정상 링크]]가 남는다.") == "앞 깨진 뒤에 정상 링크가 남는다."
    assert clean_wikitext("앞 {{깨진 틀 뒤의 본문은 유지된다.") == "앞 깨진 틀 뒤의 본문은 유지된다."
    assert "[[" not in clean_wikitext("[[a]_ 뒤 [[b]] 끝")


def test_balanced_spans_are_removed_and_nesting_respected():
    assert strip_braced("앞{{겉{{속}}겉}}뒤", "{{", "}}") == "앞뒤"
    assert strip_braced("앞{|표|}뒤", "{|", "|}") == "앞뒤"
    assert strip_links("[[파일:a.jpg|섬네일|설명]]본문[[분류:분류명]]") == "본문"
    assert strip_links("[[대상|보이는 말]]") == "보이는 말"
    assert strip_links("[[대상]]") == "대상"


def test_inequalities_survive_tag_stripping():
    """<[^>]+> would eat "< x >"; maths articles must keep their operators."""
    # Tags go, the text they wrap stays.
    assert clean_wikitext("0 < x > y 이고 <br/> <b>강조</b> 제거") == "0 < x > y 이고 강조 제거"


def test_convert_filters_namespaces_redirects_and_emits_required_fields():
    out = None
    stats = None
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)/"raw.jsonl"
        stats = convert(io.BytesIO(_bz2(DUMP)), out, "kowiki-20260901", "cc-by-sa-4.0",
                        "korean_general", max_documents=10, min_chars=10)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stats["kept"] == 1 and stats["scanned"] == 1
    row = rows[0]
    for key in ("id", "source", "license", "domain", "text"):
        assert isinstance(row[key], str) and row[key]
    assert row["id"] == "kowiki-20260901-11"
    assert row["group_id"] == "kowiki-20260901:11"
    assert row["text"].startswith("본문 문서\n\n")
    # Template, ref and the reference section are gone; the link label stays.
    assert "표시되는 말" in row["text"]
    assert "버림" not in row["text"] and "각주" not in row["text"] and "지워질" not in row["text"]


def test_convert_refuses_to_overwrite():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)/"raw.jsonl"
        target.write_text("기존", encoding="utf-8")
        with pytest.raises(ValueError):
            convert(io.BytesIO(_bz2(DUMP)), target, "s", "cc-by-sa-4.0", "korean_general", 1, 1)


def _bz2(text):
    import bz2
    return bz2.compress(text.encode("utf-8"))
