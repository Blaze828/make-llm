"""Regression cases for future execution; these do not fit a model/tokenizer."""
import json
import pytest
from data_pipeline.prepare import prepare
from data_pipeline.pack import iter_packed
from evaluation.metrics import answer_scores
from training.execution import require_training_enabled
from training.indexed_data import shuffled_index


def test_indexed_rows_reject_corruption(tmp_path):
    import numpy as np
    from data_pipeline.pack import DTYPES
    from data_pipeline.common import sha256_file, write_json
    from training.indexed_data import IndexedRows
    arrays = {}
    for key,dtype in DTYPES.items():
        file = tmp_path/f"{key}.bin"
        np.array([[1,2,3]],dtype=dtype).tofile(file)
        arrays[key] = dict(path=file.name,dtype=dtype,sha256=sha256_file(file))
    manifest = tmp_path/"manifest.json"
    write_json(manifest,dict(kind="packed_tokens",format_version=1,rows=1,sequence_length=3,arrays=arrays))
    rows = IndexedRows(manifest)
    assert rows[0]["input_ids"].tolist()==[1,2,3]
    del rows  # Close the mapped arrays before rewriting on Windows.
    (tmp_path/"input_ids.bin").write_bytes(b"\x00"*12)
    with pytest.raises(ValueError,match="checksum"): IndexedRows(manifest)


def test_long_document_retains_every_target_once():
    original = list(range(17))
    rows = list(iter_packed([original],5,99))
    pairs = [(r["input_ids"][i],r["input_ids"][i+1]) for r in rows for i in range(4)
             if r["attention_mask"][i+1] and r["document_ids"][i]==r["document_ids"][i+1]]
    assert pairs == list(zip(original,original[1:]))


def test_epoch_is_permutation_and_resume_cursor_is_stable():
    for size in (1,2,6,11,100):
        whole = [shuffled_index(i,size,42) for i in range(2*size)]
        assert sorted(whole[:size]) == list(range(size))
        assert sorted(whole[size:]) == list(range(size))
        assert whole[3:] == [shuffled_index(i,size,42) for i in range(3,2*size)]


def test_duplicate_bridge_keeps_external_groups_together(tmp_path):
    source = tmp_path/"raw.jsonl"
    records = [dict(id=str(i),text=t,group_id=g,source="fixture",license="fixture",domain="korean")
               for i,(t,g) in enumerate((("가나다 가나다", "A"),("라마바 라마바","B"),("가나다 가나다","B")))]
    source.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in records),encoding="utf-8")
    config = dict(allowed_licenses=["fixture"],min_chars=1,max_chars=100,redact_email_phone=False,
                  near_duplicate_grouping=False,near_duplicate_jaccard=.85,seed=42,validation_ratio=.2,test_ratio=.2)
    out = tmp_path/"out"
    manifest = prepare([source],out,config)
    rows = [json.loads(line) for name in ("train","validation","test")
            for line in (out/f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==2 and manifest["stats"]["exact_duplicates_removed"]==1
    assert len({r["group_id"] for r in rows})==1
    assert len({r["split"] for r in rows})==1


def test_training_gate_defaults_closed(monkeypatch):
    monkeypatch.delenv("MAKE_LLM_ALLOW_TRAINING",raising=False)
    with pytest.raises(RuntimeError): require_training_enabled()


def test_korean_character_scoring():
    assert answer_scores("서울 입니다!",["서울입니다"])["exact_match"]==1
    assert answer_scores("서울",["부산"])["character_f1"]==0
