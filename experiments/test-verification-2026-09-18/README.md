# 환경 구성과 테스트 실행 기록

실행일: 2026-09-18 · 환경: B (`inozosa`)

[2026-09-18 데이터 확보 전 준비](../data-readiness-2026-09-18/README.md)에서 추가한 코드는 정적 검사만 거친 상태였다. **이 기록은 그 코드를 처음 실제로 실행한 결과다.** 의존성을 설치하고 회귀 테스트를 돌린 것이 범위이며, 한국어 데이터 학습이나 성능 측정은 아니다.

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| VRAM | 11.99 GiB |
| Python | 3.13.15 |
| PyTorch | 2.14.0+cu126 (CUDA 사용 가능, BF16 지원) |
| numpy | 2.5.3 |
| tokenizers | 0.23.2 |
| pytest | 9.1.1 |
| mecab-ko / mecab-ko-dic | 1.0.2 / 1.0.0 |

2026-09-17 기록의 RTX 3060은 **환경 A**에서 측정한 것이며 위 수치와 직접 비교하지 않는다. 환경 간 비교 규칙은 [환경 문서](../../docs/environments.md)에 있다.

## 2. Python 3.14에서의 설치 실패와 원인

작업 시작 시점의 venv는 **Python 3.14.6**이었고 `torch`·`numpy`만 설치되어 있었다. `pytest`, `tokenizers`, `mecab-ko`가 없어 회귀 테스트를 실행할 수 없는 상태였다.

`pip install -r requirements.txt`는 `mecab-ko`에서 실패했다.

```
creating build\lib.win-amd64-cpython-314\mecab_ko
building 'mecab_ko._MeCab' extension
error: Unable to find a compatible Visual Studio installation.
ERROR: Failed building wheel for mecab-ko
```

PyPI의 `mecab-ko` 1.0.2 배포 파일을 확인한 결과 Windows wheel은 **cp313까지만** 존재하고 그 이상은 `mecab-ko-1.0.2.tar.gz` 소스뿐이었다. 따라서 Python 3.14에서는 pip이 소스 빌드로 넘어가고 C++ 확장 컴파일에 MSVC가 필요해 실패한다. 형태소 분석기를 다른 것으로 대체하지 않고 **Python 3.13으로 환경을 맞추는 방법을 선택했다.**

Python 3.13.15를 설치하고 venv를 재생성한 뒤에는 `mecab-ko`가 wheel로 설치되었다. 이전 3.14 venv는 `.venv-py314`로 남겨 두었다.

## 3. 실제 실행한 명령과 결과

| 명령 | 결과 |
|---|---|
| `python -X utf8 -m scripts.static_check` | `{"static_parsing":"passed","python":45,"json":6,"jsonl_records":10,"training_executed":false}` |
| `python -X utf8 -m pytest -q` | **15 passed, 9 skipped** |
| `MAKE_LLM_ALLOW_TRAINING=1` + `pytest -q` | **24 passed** |
| `python -X utf8 -m scripts.preflight` | `blocked`, 차단 사유 2건 |
| `python -X utf8 -m scripts.plan_budget --tokens 100000000 --batch 1 --sequence 1024 --accumulation 8` | 정상 출력 |

**실패 0건.** 2026-09-17 기록의 18개에서 24개로 늘어난 것은 `tests/test_data_pipeline.py` 6개가 추가되었기 때문이다.

기본 실행에서 건너뛴 9개는 전부 학습 게이트 때문이며 CUDA·MeCab 문제가 아니었다.

```
SKIPPED [2] tests\test_adapters.py:9: Training disabled by default
SKIPPED [2] tests\test_tokenizer.py:9: Training disabled by default
SKIPPED [1] tests\test_tokenizer.py: Training disabled by default
SKIPPED [4] tests\test_training.py: Training disabled by default
```

`test_model.py`의 `test_cuda_bf16_cache`는 CUDA가 있어 실행되었고 통과했다. CUDA 없는 환경에서는 skip되므로 통과 **개수**는 환경마다 다를 수 있다.

### MeCab 동작 확인

```
'한국어 형태소 분석기가 정상 동작한다.'
→ ['한국어', '형태소', '분석기', '가', '정상', '동작', '한다', '.']
```

사전 경로는 `.venv/Lib/site-packages/mecab_ko_dic/dicdir/sys.dic`이다. 표면형 분해만 확인했고 어휘 학습은 실행하지 않았다.

### preflight 차단 사유 (데이터 미확보 상태의 정상 결과)

```
실제 데이터 이용 조건 확인 후 allowed_licenses를 지정해야 합니다.
train/validation 토큰 데이터 manifest가 아직 지정되지 않았습니다.
```

### plan_budget 산식 결과 (측정값 아님)

`configs/architecture/base-316m.json` 기준. torch를 import하지 않고 산식만 계산한 값이며 GPU 실행 결과가 아니다.

| 항목 | 값 |
|---|---:|
| 파라미터 | 315,936,768 |
| FP32 가중치+gradient+AdamW 상태 | 4.708 GiB |
| BF16 추론 가중치 | 0.588 GiB |
| BF16 KV cache (batch 1, 길이 1024) | 0.047 GiB |
| update당 할당 입력 토큰 | 8,192 |
| 1억 토큰 최소 update 수 | 12,208 |

activation·logits·workspace 메모리가 빠져 있으므로 **11.99 GiB에서 316M 모델이 문맥 1024로 학습 가능한지는 이 값으로 판단할 수 없다.** 실측이 필요하다.

## 4. 실행하지 않은 작업

- 한국어 데이터 정제·토큰 패킹 (**데이터 없음**, `allowed_licenses` 비어 있음)
- 실제 말뭉치의 토크나이저 어휘 학습, 16K/32K/48K 비교
- 본 사전학습, SFT, 생성, 평가 harness 실행
- 환경 B의 316M 모델 GPU 메모리·처리량 측정
- 환경 A에서의 동일 테스트 재실행 (교차 확인 미완)

`MAKE_LLM_ALLOW_TRAINING=1`은 **테스트 실행에만** 사용했다. 학습 표시 테스트는 작은 토이 모델에 소수의 optimizer update를 수행하므로 파라미터가 갱신되지만, 본 학습 실행이 아니다. 이 기록에는 loss·정확도·처리량·메모리 측정 결과가 없다.

## 5. 함께 확인한 코드 사항

`training/checkpoint.py`의 `load_checkpoint`는 `run_metadata`의 완전 일치를 요구하지만, `training/pretrain.py`가 기록하는 `device`는 CLI 인자 문자열(`"cuda"`/`"cpu"`)이며 GPU 모델명이 아니다. 따라서 **GPU가 다른 머신에서 시작한 학습을 다른 머신에서 재개해도 metadata 검사를 통과한다.** 코드가 막지 않으므로 작업자가 지켜야 하며, 규칙은 [환경 문서 §5](../../docs/environments.md)에 적었다. 코드 변경은 하지 않았다.

## 6. 다음 확인이 필요한 것

1. 환경 A에서 같은 명령을 실행해 **실패 0건**과 산식 결과 일치를 확인한다. 메모리·처리량은 환경별로 따로 기록한다.
2. 환경 A의 VRAM을 확인해 [환경 문서](../../docs/environments.md) 표를 채운다.
3. 데이터 출처와 이용 조건을 확정한다 ([데이터 출처 문서](../../docs/data-sources.md)).
