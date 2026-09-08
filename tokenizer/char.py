# 텍스트를 정수 리스트로 바꾸고 다시 되돌리는 문자 단위 토크나이저

import json
import sys
from pathlib import Path

UNK = "<unk>"


class CharTokenizer:
    def __init__(self, itos):
        # itos: id를 글자로 / stoi: 글자를 id로, 서로 역방향인 두 개의 표
        self.itos = list(itos)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def train(cls, text):
        # 등장한 고유 글자를 정렬해 어휘를 만든다 
        return cls([UNK] + sorted(set(text)))

    @property
    def vocab_size(self):
        # 어휘 크기. 나중에 모델 출력층의 크기
        return len(self.itos)

    def encode(self, text):
        # 문자열 -> 정수 리스트, 모르는 글자는 UNK(0)
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids):
        # 정수 리스트 -> 문자열
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        Path(path).write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"])


SAMPLE = """자연어 처리는 사람이 쓰는 말을 컴퓨터가 다루게 하는 분야다.
언어 모델은 앞의 글자들을 보고 다음에 올 글자를 맞히는 일을 반복하며 학습한다.
이 단순한 목표 하나에서 문법과 어휘, 문맥을 파악하는 능력이 함께 자라난다.
GPT-1은 2018년에 이 방식으로 12개 과제 중 9개에서 최고 성적을 냈다."""


def selftest(text):
    # 토크나이저가 정상인지 확인
    tok = CharTokenizer.train(text)

    print(f"코퍼스     : {len(text):,} 글자")
    print(f"어휘 크기  : {tok.vocab_size:,} (고유 글자 {tok.vocab_size - 1} + UNK)")

    ids = tok.encode(text)
    assert tok.decode(ids) == text, "왕복 실패: 복원본이 원본과 다름"
    print(f"왕복 검사  : 통과 ({len(ids):,} 토큰, 한 글자도 안 틀림)")

    assert CharTokenizer.train(text).itos == tok.itos, "실행할 때마다 어휘가 달라짐"
    print("재현성     : 통과")

    path = Path("char_vocab.json")
    tok.save(path)
    reloaded = CharTokenizer.load(path)
    assert reloaded.itos == tok.itos and reloaded.decode(reloaded.encode(text)) == text
    print(f"저장/로드  : 통과 ({path}, {path.stat().st_size:,} bytes)")

    unseen = "🚀"
    assert tok.encode(unseen) == [0], "모르는 글자가 UNK로 안 감"
    print(f"UNK 처리   : 통과 ('{unseen}' -> [0])")

    s = text[:12]
    print(f"\n원본 : {s}")
    print(f"토큰 : {tok.encode(s)}")
    print(f"복원 : {tok.decode(tok.encode(s))}")

    print(f"\n압축률 : 토큰당 {len(text)/len(ids):.2f} 글자")
    return tok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        text = SAMPLE
        print("(내장 샘플 사용 - 내 파일로 하려면: python tokenizer/char.py 파일경로)\n")
    selftest(text)
