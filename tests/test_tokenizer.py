import pytest
pytestmark = pytest.mark.training
from tokenizer.bpe import BPETokenizer
from tokenizer.morph_bpe import MecabSurfaceSplitter

TEXTS = ["한국어를 잘하는 모델입니다.\n", "교차로 신호등\t대기 차량 30대 🚦", "x = 3\n    print(x)  ", "한글 English"]


@pytest.mark.parametrize("morph", [False, True])
def test_roundtrip_special_tokens_and_artifact(tmp_path, morph):
    splitter = MecabSurfaceSplitter() if morph else None
    tok = BPETokenizer.train(TEXTS, 400, splitter)
    for text in TEXTS + ["  낯선 🦄𐐀 한자漢字\r\n", "<|eos|><|assistant|>"]:
        assert tok.decode(tok.encode(text)) == text
        assert tok.special_id("<|eos|>") not in tok.encode(text)
    tok.save(tmp_path)
    restored = BPETokenizer.load(tmp_path)
    assert restored.encode(TEXTS[0]) == tok.encode(TEXTS[0])
    if morph:
        assert "".join(splitter(TEXTS[1])) == TEXTS[1]
        assert len(splitter("한국어를 잘하는 모델입니다.")) > 3
        assert tok.metadata["morphology"]["dictionary_sha256"]


def test_bad_splitter_is_rejected():
    with pytest.raises(ValueError, match="changed"):
        BPETokenizer.train(["a b"], 300, splitter=lambda text: text.split())
