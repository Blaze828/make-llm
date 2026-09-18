# tokenizer

텍스트와 정수 토큰 사이를 변환한다. (README 5번)

| 파일 | 상태 | 역할 |
|---|---|---|
| `char.py` | 완료 | 문자 단위. 파이프라인 검증용 + BPE 비교 대조군 |
| `bpe.py` | 구현 | 일반 Byte-level BPE 및 고정 vocabulary/merges 인코딩, artifact 저장·hash 검증 |
| `morph_bpe.py` | 구현 | MeCab-ko 원문 경계 기반 BPE 어휘 학습, CLI, 분석기·사전 버전 기록 |
| `benchmark.py` | 예정 | 토크나이저 비교 (압축률, 영어·숫자·혼합·전문용어·신조어 처리) |

모든 토크나이저는 같은 인터페이스를 따른다.
`encode(text) -> list[int]` / `decode(ids) -> str` / `vocab_size` / `save(path)` / `load(path)`

이 규약 덕분에 model, training 코드는 토크나이저 종류를 몰라도 된다.

현재 설계는 [한국어 tokenizer 계약](../docs/korean-model-recipe.md)을 따른다. 형태소 분석은 기본적으로 **어휘 학습 때만** 수행한다. 모델 사전학습·미세조정·추론에서는 같은 ByteLevel 인코딩을 쓴다. 원문 공백·개행·코드를 보존하고 모든 byte alphabet을 포함한다. 완성 모델의 tokenizer를 변경하면 checkpoint와 adapter를 새 버전으로 다뤄야 한다.
