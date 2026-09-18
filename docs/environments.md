# 실행 환경과 결과 비교 규칙

2026-09-18 작성. 이 프로젝트는 **GPU가 서로 다른 여러 머신에서 작업한다.** 같은 코드·같은 seed라도 환경에 따라 달라지는 값이 있으므로, 어떤 결과를 환경 간에 비교할 수 있고 어떤 결과를 비교할 수 없는지 여기서 고정한다.

실험 기록에 환경을 적지 않은 수치는 다른 작업자가 재현 기준으로 사용할 수 없다.

## 1. 확인된 실행 환경

| 환경 | 작업자 | GPU | VRAM | Python | PyTorch | 확인일 |
|---|---|---|---|---:|---|---|
| A | 유상진 `samgjim@naver.com` | NVIDIA GeForce RTX 3060 | 기록 없음 | 3.13.12 | 2.11.0+cu126 | 2026-09-17 |
| B | inozosa `inozosa0102@gmail.com` | NVIDIA GeForce RTX 4070 SUPER | 11.99 GiB | 3.13.15 | 2.14.0+cu126 | 2026-09-18 |

환경 A의 VRAM은 당시 기록에 없다. RTX 3060은 12GB·8GB 모델이 모두 존재하므로 **추정하지 않는다.** 환경 A에서 작업할 때 `torch.cuda.get_device_properties(0).total_memory`로 확인해 이 표를 채운다.

기존 실험 기록([2026-09-17 검증](../experiments/implementation-validation-2026-09-17/README.md))의 GPU 수치는 **환경 A에서 측정한 값이다.** 특히 316M 모델의 최대 CUDA allocated 메모리 약 5.84 GiB는 환경 A, batch 1, 길이 32 기준이며 다른 환경·다른 설정의 예측값이 아니다.

### 검증 기준선

환경 B에서 2026-09-18 확인한 값이다. 실행 상세는 [환경 구성·테스트 기록](../experiments/test-verification-2026-09-18/README.md)에 있다. 다른 환경에서도 **실패 0건**이 나와야 한다.

| 명령 | 환경 B 결과 |
|---|---|
| `python -X utf8 -m scripts.static_check` | `passed` (python 45, json 6, jsonl 10) |
| `python -X utf8 -m pytest -q` | `15 passed, 9 skipped` |
| `MAKE_LLM_ALLOW_TRAINING=1` + `pytest -q` | `24 passed` |
| `python -X utf8 -m scripts.preflight` | `blocked` (데이터 미확보 상태의 정상 결과, 종료 코드 2) |

건너뛴 9개는 모두 학습 게이트 때문이며 CUDA·MeCab 문제가 아니다. 게이트를 열면 24개가 모두 실행된다. 2026-09-17 기록의 18개보다 6개 늘어난 것은 `tests/test_data_pipeline.py`가 추가되었기 때문이다.

### 환경 B 실측값 (2026-09-18 파일럿)

`debug-40m`(39,986,688 파라미터), 문맥 1,024, micro batch 4 × accumulation 8, BF16, activation checkpointing 미사용.

| 항목 | 값 |
|---|---|
| `tokens_per_second` 중앙값 | 약 17,000 (범위 11,915~19,730) |
| GPU 메모리 (프로세스) | 8,040 MiB / 12,282 MiB |
| `data_pipeline.prepare` 처리량 | 57MB를 5분 11초 (약 184KB/s) |

출처와 전체 조건은 [파일럿 기록](../experiments/pilot-kowiki-2026-09-18/README.md)에 있다. **환경 A에서는 다르게 측정된다.** `base-316m`의 실측값은 아직 없다.

## 2. 환경이 달라도 같아야 하는 것

아래가 환경마다 다르면 **코드 결함이지 환경 차이가 아니다.** 결과가 갈리면 환경 탓으로 넘기지 말고 원인을 찾는다.

- `pytest` 통과/실패 여부와 실패한 테스트 이름
- 문서 격리 packing, padding 누출 없음, KV cache 일치, eager/SDPA 출력·gradient 비교
- 세 아키텍처 설정의 파라미터 수, `scripts/plan_budget.py`의 산식 결과
- tokenizer의 `decode(encode(text))` 복원과 special token 격리 (byte-level 결정적 연산)
- manifest·checksum·vocab 크기·문맥 길이 검증의 통과/거절 판정
- CPU 경로의 저장·재개 동일성, gradient accumulation과 full batch의 일치
- `scripts/static_check.py`, `scripts/preflight.py`의 판정

## 3. 환경에 따라 반드시 달라지는 것

아래는 환경 차이가 정상이다. **다른 환경의 값을 자기 환경 계획에 그대로 옮겨 쓰지 않는다.**

| 항목 | 이유 |
|---|---|
| 최대 CUDA allocated / reserved 메모리 | SM 수, allocator 분할, 커널 workspace가 GPU마다 다름 |
| `tokens_per_second`, 학습 소요 시간 | 연산 성능과 메모리 대역폭 차이 |
| OOM 발생 여부 | VRAM 용량 차이. 한 환경에서 도는 batch·sequence·`--gradient-checkpointing` 설정이 다른 환경에서 실패할 수 있다 |
| BF16 학습의 CE 하위 자리 | BF16 CUDA 연산의 bitwise 재현성을 보장하지 않는다 ([구현 문서](implementation.md) 참조). 같은 seed라도 마지막 자리가 어긋날 수 있다 |
| `pytest` 통과/건너뜀 **개수** | `test_cuda_bf16_cache`는 CUDA 없는 환경에서 skip된다. 개수가 아니라 실패 0건을 기준으로 삼는다 |

## 4. 규칙

1. **측정값에는 환경을 함께 적는다.** 메모리·처리량·소요 시간을 기록할 때 GPU 모델, VRAM, Python, PyTorch 버전, batch/sequence/accumulation, precision을 같은 표에 적는다. 환경 표기가 없는 수치는 실험 기록으로 인정하지 않는다.
2. **`--measured-tokens-per-second`에는 자기 환경 실측값만 넣는다.** `scripts/plan_budget.py`의 시간 추정은 입력한 처리량에 그대로 비례한다. 다른 환경 값을 넣으면 학습 기간을 잘못 산정한다.
3. **VRAM이 작은 환경을 기준으로 설정을 정한다.** 한 환경에서만 도는 설정을 본 학습 기본값으로 쓰면 다른 작업자가 실행할 수 없다. 환경별 실행 가능 설정을 따로 기록한다.
4. **학습 중단·재개는 같은 환경에서 한다.** 아래 §5 참조.
5. **손실 곡선을 환경 간에 비교할 때는 같은 device·precision을 쓴다.** BF16 CUDA끼리도 완전 일치를 기대하지 않는다. 비교가 목적이면 FP32 또는 CPU 경로로 맞춘다.
6. **pytest 결과를 공유할 때는 `15 passed, 9 skipped` 같은 숫자와 함께 환경을 적는다.**

## 5. 재개에 관한 알려진 허점

`training/checkpoint.py`의 `load_checkpoint`는 `run_metadata`의 **완전 일치**를 요구한다. 그런데 `training/pretrain.py`가 넣는 `device` 값은 CLI 인자 문자열(`"cuda"` 또는 `"cpu"`)이며 **GPU 모델명이 아니다.**

따라서 환경 A에서 `--device cuda`로 시작한 학습을 환경 B에서 `--device cuda`로 재개하면 **metadata 검사를 그대로 통과한다.** 코드가 막아주지 않으므로 작업자가 지켜야 한다:

- 한 번 시작한 학습 run은 **같은 머신에서 끝낸다.**
- 머신을 옮겨야 하면 동일 run 재개로 취급하지 말고, 옮긴 사실과 시점(step)을 실험 기록에 남긴다.
- 옮긴 run의 손실 곡선을 하나의 연속된 실행으로 보고하지 않는다.

`cuda_rng`는 `get_rng_state_all()`로 저장한다. GPU 개수가 다른 환경으로 옮기면 RNG 상태 복원이 의도와 달라질 수 있다. 현재 두 환경 모두 GPU 1개다.

## 6. Python 버전 고정 (3.13)

**Python 3.13을 사용한다. 3.14를 쓰면 안 된다.**

`mecab-ko==1.0.2`는 PyPI에 **cp313까지만** wheel을 올려 두었다. Python 3.14에서는 pip이 `mecab-ko-1.0.2.tar.gz` 소스 빌드로 넘어가고, C++ 확장 컴파일에 Visual Studio가 필요해 다음 오류로 실패한다.

```
error: Unable to find a compatible Visual Studio installation.
ERROR: Failed building wheel for mecab-ko
```

`requirements.txt`는 Python 버전 제약을 표현할 수 없으므로 여기에 기록한다. venv를 만들 때 3.13 인터프리터를 명시한다.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

MeCab은 **토크나이저 어휘 학습 단계에서만** 쓴다. 모델 학습과 추론은 byte-level이라 MeCab 없이도 동작한다. 형태소 없는 byte BPE(`--plain`)는 비교용 대조군이며 MeCab 설치 실패의 대체 경로가 아니다.
