# 데이터 확보 전 준비 — 2026-09-18

이번 작업은 코드 작성과 정적 검사만 진행했다. 모델 학습, 토크나이저 어휘 학습, 추론, 평가, pytest를 실행하지 않았다. 아래 명령은 향후 사용법이다. 기존 실험 결과는 [별도 기록](../experiments/implementation-validation-2026-09-17/README.md)에 있으며 이번 코드의 동작 검증 결과가 아니다.

## 준비한 기능

| 단계 | 구현 | 내용 |
|---|---|---|
| 원문 준비 | `data_pipeline/prepare.py` | 출처·이용 조건 검사, HTML 처리, 이메일·휴대전화 패턴 치환, 깨진 텍스트·반복 필터 |
| 중복과 분할 | 동일 모듈 | NFC/공백 정규화 기반 exact dedup, MinHash 후보의 Jaccard 검증, 관련 문서 그룹 단위 train/validation/test 분할 |
| 토큰 데이터 | `data_pipeline/pack.py` | 학습된 tokenizer로 순차 변환, 문서 경계·padding 기록, 긴 문서 한 토큰 overlap, binary 배열·checksum |
| 데이터 로딩 | `training/indexed_data.py` | memmap 행 로딩, 파일 검증, epoch별 결정적 순열 |
| 학습 운용 | `training/pretrain.py` | CPU microbatch 대기열, 선택적 activation checkpointing, JSONL 로그, 주기·최상·초기 저장, 장애 상태 기록 |
| 토크나이저 비교 | `tokenizer/benchmark.py` | 기존 artifact별 분야별 tokens/byte·원문 복원 실패·인코딩 처리량 |
| 한국어 평가 | `evaluation/run.py` | 분야별 CE/PPL/BPB, 객관식 continuation likelihood, QA EM·문자 F1, 생성 반복·EOS 비율 |
| 분야 적응 비교 | `evaluation/compare.py` | 같은 평가 문항·프로토콜의 base/adapter 결과 차이 |
| 실행 전 점검 | `scripts/preflight.py` | 데이터 artifact·어휘 크기·문맥·checksum 점검, 부족한 입력 표시 |
| 예산 계산 | `scripts/plan_budget.py` | 파라미터·학습 상태 메모리·KV cache·토큰/update·대략적인 FLOPs 계산 |
| 실행 차단 | `training/execution.py` | 학습 opt-in 환경변수 검사, 지원하지 않는 다중 프로세스 실행 거절 |

모델 구조는 Dense decoder-only + GQA + QK-RMSNorm + RoPE + SwiGLU + tied embedding을 유지했다. 이 조합이 한국어에 최적이라는 결론은 아직 낼 수 없다. 한국어 효율은 실제 말뭉치로 토크나이저와 모델을 비교해야 확인할 수 있다.

## 1. 원문 데이터 형식

UTF-8 JSONL, 한 줄에 한 문서. 문서 내부 줄바꿈은 JSON의 `\n`으로 보존한다.

```json
{"id":"source-0001","source":"확인한 원본 출처","license":"확인한 이용 조건 식별자","domain":"korean_general","text":"문서 내용\n다음 문단","group_id":"같은 원문이나 시리즈 식별자","content_type":"text"}
```

`id/source/license/domain/text`는 필수다. `group_id`는 선택이며 같은 원문에서 파생한 여러 문서를 묶을 때 사용한다. 서로 다른 출처의 group ID가 충돌하지 않도록 출처 접두사를 붙인다. HTML 문서는 `content_type=html`을 쓴다. 코드 분야에는 HTML 제거를 적용하지 않는다.

`configs/data/preparation.json`의 `allowed_licenses`는 의도적으로 비워 두었다. 실제 자료의 조건을 확인한 후 허용 식별자를 넣어야 실행된다. 이름을 허용 목록에 넣는 것 자체가 이용 권리를 증명하지는 않는다.

```powershell
python -X utf8 -m data_pipeline.prepare --input datasets/raw.jsonl --output datasets/clean-v1
```

결과: `train.jsonl`, `validation.jsonl`, `test.jsonl`, `rejected.jsonl`, `manifest.json`, `dedup.sqlite`. 출처·분야별 문서 수, split별 byte 수, 입력·출력 hash와 설정을 기록한다. 완전 중복은 제거하고 탐지한 유사 문서는 같은 split에 유지한다. 기본 분할 비율 96/2/2는 그룹 hash 기준 기대 비율로, 정확한 문서 수를 보장하지 않는다. 작은 코퍼스는 빈 split이 생길 수 있다.

MinHash 후보 검색은 유사 문서를 놓칠 수 있고 개인정보 치환은 일부 패턴만 다룬다. 광고·문서 품질·권리·외부 benchmark 오염 검토는 실제 데이터를 확보한 후 필요하다. SQLite 정제는 디스크를 사용하지만 큰 문서의 shingle 집합과 후보 집합은 메모리를 사용한다. 대규모 처리 속도는 측정하지 않았다.

## 2. 토크나이저와 토큰 데이터

`tokenizer.morph_bpe --corpus-manifest datasets/clean-v1/manifest.json` 경로를 추가했다. train split만 어휘 학습에 사용하고 원본 hash를 확인한다. 기존 `--text`도 지원한다. **어휘 학습 역시 학습이므로 현재 실행하지 않는다.** 향후 16K/32K/48K 및 형태소 유무를 비교하고 실제 어휘 크기를 모델 config에 맞춘다.

학습된 tokenizer가 준비된 이후에 실행할 명령:

```powershell
python -X utf8 -m tokenizer.benchmark --tokenizer morph32=checkpoints/korean-tokenizer --input datasets/clean-v1/validation.jsonl --output experiments/tokenizer-comparison.json
python -X utf8 -m data_pipeline.pack --corpus-manifest datasets/clean-v1/manifest.json --split train --tokenizer checkpoints/korean-tokenizer --sequence-length 1024 --output datasets/tokens-v1/train
python -X utf8 -m data_pipeline.pack --corpus-manifest datasets/clean-v1/manifest.json --split validation --tokenizer checkpoints/korean-tokenizer --sequence-length 1024 --output datasets/tokens-v1/validation
```

비교할 tokenizer는 `--tokenizer 이름=폴더`를 반복한다. 같은 held-out 문서와 환경으로 비교한다. `evaluation/examples/tokenizer-cases.jsonl`은 한영 혼합·자모·공백·special token 문자열의 직접 작성 예시이며 성능 benchmark가 아니다. 토크나이저 결정에는 validation만 쓰고 test는 마지막 평가까지 보관한다.

패킹은 문서 사이의 attention과 loss를 분리하고 긴 문서의 다음 토큰 target이 경계에서 사라지지 않게 한 토큰을 겹친다. 한 문서 토큰화 결과와 한 행을 메모리에 보관한다. 디스크 행은 input ID 4byte, document ID 8byte, mask 1byte로 토큰 위치당 13byte이며 패딩도 공간을 차지한다.

## 3. 실행 전 점검과 예산

```powershell
python -X utf8 -m scripts.static_check
python -X utf8 -m scripts.preflight --train-manifest datasets/tokens-v1/train/manifest.json --validation-manifest datasets/tokens-v1/validation/manifest.json
python -X utf8 -m scripts.plan_budget --tokens 100000000 --batch 1 --sequence 1024 --accumulation 8
```

예산 명령의 1억 토큰은 사용법 예시이며 확정 목표가 아니다. GPU 모델을 만들거나 학습하지 않고 산식만 계산한다. AdamW 상태 추정은 FP32 가중치·gradient·두 momentum의 합이며 activation/logits/workspace 메모리는 제외한다. 시간은 사용자가 측정한 valid target tokens/sec를 입력한 경우에만 계산한다. 단순 token/update 산식은 padding·문서 경계 손실을 제외한 하한 추정이다.

preflight는 준비가 부족하면 종료 코드 2를 반환한다. artifact 검사 통과는 실행 성능이나 학습 안정성을 보장하지 않는다.

## 4. 향후 학습 실행 경로

현재는 `MAKE_LLM_ALLOW_TRAINING=1` 없이는 모델·토크나이저 학습 진입점이 중단된다. 이 변수는 이번 작업에서 설정하지 않았다. 나중에 학습을 하기로 결정했을 때 해당 세션에서 명시적으로 설정해야 한다. default pytest는 학습 표시 테스트를 건너뛰지만 모델 계산이 있는 다른 테스트는 실행하므로 이번에는 pytest도 실행하지 않았다.

```powershell
python -X utf8 -m training.pretrain --train-manifest datasets/tokens-v1/train/manifest.json --validation-manifest datasets/tokens-v1/validation/manifest.json --tokenizer checkpoints/korean-tokenizer --model-config configs/architecture/base-316m.json --steps 1000 --batch-size 1 --accumulation 8 --sequence-length 1024 --device cuda --bf16 --gradient-checkpointing --save-every 100 --eval-every 100 --output checkpoints/base-v1.pt
```

위 값은 CLI 연결 예시이며 GPU 적합성을 검증한 설정이 아니다. validation은 현재 FP32 forward를 사용하므로 BF16 학습보다 추가 메모리가 필요할 수 있다. 단일 장치만 지원한다. affine sampler는 epoch마다 모든 행을 한 번씩 방문하지만 모든 순열 중 균등하게 뽑는 완전 무작위 shuffle은 아니다.

체크포인트 외에 `.events.jsonl`, `.status.json`, `.best.pt`를 남긴다. 로그에는 CE·Z-loss·실제 사용 LR·다음 LR·gradient norm·유효 token 수·속도를 기록한다. 장애가 나면 마지막 **저장된** checkpoint부터 재개하므로 저장 간격 내 완료 update는 다시 수행할 수 있다. 진행 중 실패한 optimizer 상태를 정상 checkpoint에 덮어쓰지 않는다. 재개 명령은 원래 조건을 유지하고 `--resume checkpoints/base-v1.pt`를 추가한다. 새 metadata와 sampler를 도입했으므로 과거 버전 실행의 동일 run 재개는 지원하지 않는다.

SFT는 기존 LoRA/DoRA 경로를 사용한다. SFT optimizer 중간 재개, 분산 학습, QLoRA/MoE는 이번 준비 범위에 포함하지 않았다.

## 5. 평가와 비교

`evaluation/examples/suite.jsonl`은 네 가지 입력 형식만 보여주는 직접 작성 예시다. 실제 KOBEST/KorQuAD 데이터도 아니고 성능을 주장할 근거도 아니다. 공식 평가가 필요하면 해당 데이터·prompt·공식 scorer를 별도로 연결해야 한다.

```powershell
python -X utf8 -m evaluation.run --checkpoint checkpoints/base-v1.pt --tokenizer checkpoints/korean-tokenizer --suite datasets/evaluation.jsonl --output experiments/base-eval --device cuda
python -X utf8 -m evaluation.run --checkpoint checkpoints/base-v1.pt --tokenizer checkpoints/korean-tokenizer --adapter checkpoints/traffic-dora --suite datasets/evaluation.jsonl --output experiments/adapter-eval --device cuda
python -X utf8 -m evaluation.compare --before experiments/base-eval/results.json --after experiments/adapter-eval/results.json --output experiments/domain-comparison.json
```

QA는 NFC·소문자화·공백/문장부호 제거 후 EM/문자 F1을 계산한다. 객관식은 prompt와 선택지를 각각 인코딩한 continuation의 총 log probability를 사용하며 길이 정규화하지 않는다. 문서 likelihood는 긴 문서를 한 토큰 겹쳐 나눠 평가하므로 경계에서 긴 문맥을 유지하지 않는다. tokenizer가 다르면 token CE/PPL을 직접 비교하지 않고 BPB를 참고한다. 서로 다른 tokenizer에서 동일 max-new-tokens는 동일 출력 byte 예산이 아니라는 점도 고려해야 한다.

## 데이터가 있어야 마무리할 사항

실제 출처·이용 조건, 한국어/영어/코드/분야 혼합 비율, 평가 오염 검사, 토크나이저 선택, 총 token 예산과 GPU 실행 설정은 아직 미정이다. 새 코드의 실행·수치 검증도 남아 있다. 이번 변경으로 이 작업을 진행할 코드와 형식을 준비했으며, 학습 품질이나 대규모 처리 안정성을 검증했다고 주장하지 않는다.
