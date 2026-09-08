# tokenizer

텍스트와 정수 토큰 사이를 변환한다. (README 5번)

| 파일 | 상태 | 역할 |
|---|---|---|
| `char.py` | 완료 | 문자 단위. 파이프라인 검증용 + BPE 비교 대조군 |
| `bpe.py` | 예정 | 바이트 레벨 BPE. 16K / 32K / 48K 세 가지 학습 |
| `benchmark.py` | 예정 | 토크나이저 비교 (압축률, 영어·숫자·혼합·전문용어·신조어 처리) |

모든 토크나이저는 같은 인터페이스를 따른다.
`encode(text) -> list[int]` / `decode(ids) -> str` / `vocab_size` / `save(path)` / `load(path)`

이 규약 덕분에 model, training 코드는 토크나이저 종류를 몰라도 된다.
