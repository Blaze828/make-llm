# 구현과 실행

2026-09-18 갱신. 설계 문서를 실제 PyTorch 코드로 구현했다. **2026-09-17 버전의 실행 기록과 이후 추가 코드의 검증 범위를 구분한다.** 데이터 파이프라인·평가·학습 보강 코드는 정적 검사만 수행했으며, 한국어 본 학습 모델은 아직 없다. 추가 범위와 사용법은 [데이터 확보 전 준비](data-readiness.md)를 참고한다.

## 구현 범위

| 구성 | 구현 위치 | 동작 |
|---|---|---|
| 구조 설정 | `model/config.py` | 기존 세 JSON 로드, 차원·지원 옵션 검증, 파라미터 산식 |
| 모델 본체 | `model/transformer.py` | embedding 공유, 순차 residual, generation |
| 블록 | `model/attention.py`, `norm.py`, `rope.py`, `mlp.py` | GQA, headwise QK-RMSNorm, RoPE, SwiGLU |
| 추론 cache | `model/cache.py` | KV heads만 저장, 모델/adapter 변경 시 무효화 |
| 한국어 tokenizer | `tokenizer/bpe.py`, `morph_bpe.py` | MeCab-ko 경계 기반 어휘 학습, byte BPE 인코딩, hash 검증 |
| 학습 | `training/pretrain.py`, `engine.py` | 단일 장치 FP32/BF16, 누적 gradient, 학습·검증 CLI |
| Loss·최적화 | `training/loss.py`, `optimizer.py` | CE·선택적 Z-loss, AdamW, no-decay 그룹, warmup/cosine |
| 저장·재개 | `training/checkpoint.py` | 모델·optimizer·scheduler·RNG·data cursor·데이터/tokenizer hash |
| 분야 적응 | `adapters/lora.py`, `training/finetune.py` | LoRA/DoRA, 응답 위치만 SFT loss, adapter 호환성 검사 |
| 생성 | `training/generate.py` | base 또는 adapter, 일반 prompt 또는 SFT 대화 포맷 |

**Python 3.13을 사용한다.** `mecab-ko==1.0.2`는 cp313까지만 wheel을 제공하므로 3.14에서는 소스 빌드로 넘어가 실패한다. GPU가 다른 환경 사이에서 무엇을 비교할 수 있는지는 [환경 문서](environments.md)에 정리했다.

MeCab은 `mecab-ko==1.0.2`와 `mecab-ko-dic` 한국어 사전을 사용한다. `python-mecab-ko` 배포판은 현재 Windows/Python 3.13에서 헤더 부족으로 빌드되지 않아, 같은 MeCab-ko 엔진의 Windows wheel 제공 배포판을 사용했다. 다른 형태소 분석기로 대체하지 않았다. 실제 버전·사전 hash는 tokenizer metadata에 기록한다.

## 설치와 전체 검증

아래는 향후 실행 참고이며 이번 작업에서는 실행하지 않았다. 저장소 루트에서 실행한다. CUDA 학습을 하려면 GPU에 맞는 PyTorch 설치가 필요하다. 학습 진입점은 `MAKE_LLM_ALLOW_TRAINING=1`을 명시하기 전까지 오류로 중단된다. 기본 pytest는 `training` 표시 테스트를 건너뛴다. 현재 코드만 확인하려면 `python -X utf8 -m scripts.static_check`를 사용한다.

```powershell
python -m pip install -r requirements.txt
python -X utf8 -m pytest -q
python -X utf8 -m scripts.smoke --device cuda --bf16 --steps 40 --output checkpoints/my-smoke
```

CPU에서는 `--device cpu`로 바꾸고 `--bf16`을 뺀다. smoke는 직접 작성한 작은 한국어 예문으로 tokenizer부터 학습·검증·저장·생성까지 연결한다. 기존 결과 덮어쓰기를 피하도록 비어 있는 output 폴더만 받는다. 약 126K 모델로 실행되며, 결과는 반복·공백 등 무의미한 생성일 수 있다. 품질 benchmark가 아니다.

## 실제 train 파일로 토크나이저 학습

```powershell
python -X utf8 -m tokenizer.morph_bpe --text datasets/train.txt --output checkpoints/korean-tokenizer
```

UTF-8 텍스트의 한 줄을 한 문서로 처리한다. 여러 줄 문서는 지금 CLI에서 하나의 문서로 묶이지 않는다. 어휘 학습에는 train만 사용한다. 모델 학습과 추론에서는 MeCab을 호출하지 않는다. `--plain`은 형태소 없는 byte BPE 대조군이며 자동 fallback 옵션이 아니다.

작은 말뭉치에서는 32K만큼 merge를 만들지 못할 수 있다. `metadata.json`의 **actual_vocab_size**가 실제 크기이며, 모델 config의 vocab_size를 이 값에 맞춰야 한다. 큰 기본 config로 본 학습할 때는 목표 32K에 도달할 만큼 데이터가 있어야 한다. 실행기는 둘이 다르면 오류를 내며 임의로 ID나 embedding을 채우지 않는다.

special token ID는 recipe의 목록 순서대로 예약한다. 일반 텍스트의 `<|eos|>` 같은 문자열은 실제 종료 명령으로 인코딩하지 않는다. 원문 whitespace·UTF-8 byte 표현을 보존한다.

## 사전학습과 재개

```powershell
python -X utf8 -m training.pretrain --text datasets/train.txt --validation-text datasets/validation.txt --tokenizer checkpoints/korean-tokenizer --model-config configs/architecture/debug-40m.json --steps 1000 --batch-size 1 --accumulation 4 --sequence-length 128 --device cuda --bf16 --output checkpoints/base.pt
```

위 text 모드는 작은 데이터용이며 정확히 같은 train/validation 문서만 추가 검사한다. 새 `data_pipeline.prepare`는 정제·exact dedup·MinHash 후보의 유사도 검증·그룹 분할을 지원한다. 대규모 입력은 `--train-manifest`와 `--validation-manifest` 디스크 로딩 경로를 사용한다. 예시는 실행 방법이지 권장 본 학습 예산은 아니다.

학습 recipe에서 optimizer·scheduler·Z-loss·clipping을 읽는다. 장치·정밀도·배치·누적 횟수·총 update 수는 CLI 인자가 실제 실행값이다. recipe에 남은 null 예산을 자동 결정하지 않는다. `--steps`는 총 optimizer update 수다. epoch별 affine 순열로 행을 방문하며 cursor로 재개한다. 이는 균등 무작위 shuffle이 아니며 분산 sampler도 아니다.

중간 종료를 지정하려면 같은 명령에 `--stop-after 100`을 붙인다. 재개할 때는 이 옵션을 빼고 `--resume checkpoints/base.pt`를 추가한다. `--steps` 등 원 실행 조건은 유지해야 한다. 스케줄·토크나이저·데이터가 바뀌면 동일 run 재개를 거절한다. `--save-every` 간격과 정상 종료 때 저장하며 초기 checkpoint와 validation 최저 CE checkpoint도 남긴다. 이전 버전 CPU 재개 테스트는 통과했지만 **이번 sampler·로그·checkpoint 변경은 아직 실행 검증하지 않았다.** 이전 실행 기록에는 새 metadata가 없으므로 새 CLI의 동일 run 재개 대상으로 호환되지 않는다. CUDA bitwise 재현성은 보장하지 않는다.

현재 pretrain checkpoint 형식은 base 모델용이다. adapter가 붙은 모델의 전체 학습 재개 포맷으로 쓰지 않는다.

## LoRA / DoRA 분야 SFT

train/validation을 각각 아래 JSONL 형식으로 준비한다. 두 파일의 prompt가 중복되면 거절한다.

```json
{"prompt":"빨간 신호에서는 어떻게 하나요?","response":"멈추고 주변을 살핍니다."}
```

```powershell
python -X utf8 -m training.finetune --base checkpoints/base.pt --tokenizer checkpoints/korean-tokenizer --train datasets/sft-train.jsonl --validation datasets/sft-validation.jsonl --method dora --rank 8 --alpha 16 --steps 100 --sequence-length 128 --device cuda --bf16 --output checkpoints/traffic-dora
```

`--method lora`로 기준선을 비교한다. base를 고정하고 adapter만 업데이트하며 prompt는 loss에서 제외한다. 문맥을 넘는 예제는 응답을 몰래 자르지 않고 오류를 낸다. adapter는 base weight·config·tokenizer·dtype hash/값이 맞아야 로드된다. adapter 파일 저장·재로딩은 지원하며 **SFT optimizer 중간 재개 CLI는 아직 없다**.

```powershell
python -X utf8 -m training.generate --checkpoint checkpoints/base.pt --tokenizer checkpoints/korean-tokenizer --adapter checkpoints/traffic-dora --chat --prompt "교차로 상황을 설명해줘." --max-new-tokens 32 --device cuda
```

`--adapter`와 `--chat`을 빼면 기본 모델의 일반 텍스트 이어쓰기다. 출력은 새로 생성한 토큰 부분이다.

## 2026-09-17 버전의 실제 검증 결과

- `pytest`: 18개 테스트 통과. causality, packing/padding 격리, cache 일치, eager/SDPA 출력·gradient 비교, 세 모델의 파라미터 수, gradient accumulation, 저장/재개, 문자 복원, adapter 저장/복원, 응답 전용 SFT loss 포함.
- RTX 3060 / PyTorch 2.11.0+cu126에서 BF16 cache 테스트 통과. (환경 A 측정값 — [환경 문서](environments.md) 참조)
- 형태소 BPE + 약 126K smoke 모델 40 update 학습·별도 문장 validation·checkpoint·생성 CLI 완료. 생성은 공백 반복 수준으로, 한국어 성능의 증거가 아니다.
- 같은 smoke base의 DoRA SFT 3 update와 별도 prompt validation 완료.
- 실제 **315,936,768 파라미터** 기본 모델: BF16, batch 1, 길이 32, AdamW 1 update 완료. 최대 CUDA allocated 메모리 약 **5.84 GiB**(환경 A 기준이며 다른 GPU의 예측값이 아니다). 4K 문맥 VRAM 추정이나 장시간 안정성 측정으로 해석하지 않는다.

## 현재 한계

단일 장치 연구용 구현이다. manifest 경로는 memmap으로 필요한 행을 읽고 text/SFT 경로는 전체 데이터를 메모리에 올린다. memmap 시작 시 전체 파일 checksum을 읽는 I/O 비용이 있다. attention은 명시적 문서 마스크와 임시 KV head 반복을 사용하며 최적 kernel 구현은 아니다. GQA cache 자체는 압축된 head 수를 유지한다. activation checkpointing은 선택적으로 지원하지만 명시적 문서 마스크의 제곱 메모리 비용까지 제거하지 않는다.

분산 학습, QLoRA 4-bit backend, p-tuning, 하이브리드/MoE, 데이터 자동 수집, 공식 KOBEST 전체 평가 harness는 구현하지 않았다. 새 평가는 사용자 JSONL 기반 일반 도구이며 공식 benchmark 점수를 대체하지 않는다. 한국어 본 학습용 데이터와 예산도 아직 확정되지 않았다.
