# 구현 검증 실행 기록

실행일: 2026-09-17 · 사후 정리일: 2026-09-18.

이 기록은 이전에 실행한 결과를 저장된 checkpoint 및 당시 대화의 콘솔 출력에서 정리한 것이다. **이번 정리에서는 학습·추론·평가를 재실행하지 않았다.** 전체 stdout 로그를 파일로 보관하지 않았으므로 아래 결과는 전체 로그가 아닌 확인 가능한 주요 수치다. 학습된 한국어 모델의 품질 검증이 아니라 구현 동작 확인이었다.

## 1. 작은 기본 모델

| 모델 항목 | 실제 값 |
|---|---:|
| 구조 | Dense Decoder-only Transformer |
| 총 학습 파라미터 | 126,400 |
| Decoder layer | 2 |
| Hidden size | 64 |
| Q / KV heads | 4 / 2 |
| Head dimension | 16 |
| SwiGLU intermediate size | 192 |
| Vocabulary | 목표 512, 실제 433 |
| 모델 최대 문맥 | 256 |
| 실제 학습 sequence length | 128 |
| Normalization | Pre-RMSNorm + headwise QK-RMSNorm |
| RMSNorm epsilon | 1e-6 |
| RoPE theta / 회전 비율 | 10,000 / 전체 head dimension |
| Attention backend | PyTorch SDPA |
| Attention / hidden dropout | 0 / 0 |
| Bias | 없음 |
| Embedding / LM Head | 가중치 공유 |
| 초기화 | 평균 0, 표준편차 0.02 정규분포; norm scale 1 |

| 학습 항목 | 실제 값 |
|---|---|
| 장치 | NVIDIA GeForce RTX 3060, CUDA |
| 실행 환경 | Python 3.13.12, PyTorch 2.11.0+cu126 |
| Train / validation | 직접 작성한 한국어 6문서 / 별도 1문서 |
| Optimizer updates | 40 |
| Microbatch / accumulation / GPU 수 | 1 / 1 / 1 |
| Seed | 42 |
| Precision | BF16 autocast, FP32 master parameter·loss·optimizer state |
| AdamW peak LR | 3e-4 |
| Betas / epsilon | [0.9, 0.95] / 1e-8 |
| Weight decay | projection 0.1; embedding·공유 head·norm scale 0 |
| Schedule | warmup 1% 설정 → cosine, 최소 LR 비율 0.1 |
| 실제 warmup | 코드의 최소값 때문에 1 update; 첫 update부터 peak LR |
| 종료 LR | 약 3e-5 |
| Global gradient clip | 1.0 |
| Z-loss coefficient | 0: CE만 최적화, Z 통계는 별도 계산 |
| Data order | 고정 순서 반복, 문서 격리 packing |

128칸짜리 packed row 두 개를 번갈아 사용했고 유효 next-token target은 각각 103개·28개였다. 40 update의 누적 유효 target은 **2,620개**(반복 포함)다. 고유 데이터 token 수나 대규모 사전학습량을 뜻하지 않는다.

| 당시 결과 | 수치 |
|---|---:|
| 첫 update train CE | 6.1162156781 |
| 마지막 update train CE | 5.0698531015 |
| Validation CE | 5.8321235657 |
| Validation perplexity | 341.0822210520 |
| Validation 유효 target 수 | 40 |

첫 update와 마지막 update는 서로 다른 packed row이므로 두 CE의 차이를 동일 평가셋의 개선으로 해석하지 않는다. validation은 학습 종료 후 값만 있고 초기 validation 측정은 없다.

생성은 greedy(temperature=0), 최대 새 token 12개, prompt `한국어 모델은`이었다. 당시 출력은 prompt 뒤 공백 반복이었다. 문장 생성 능력을 획득했다고 볼 수 없다.

토크나이저는 MeCab-ko 1.0.2, mecab-ko-dic 1.0.0, tokenizers 0.23.2로 학습했다. 형태소는 어휘 학습 때만 사용하고 실제 모델 입력은 ByteLevel 인코딩이다. 분석 fallback 문서는 0개였다. 실제 vocabulary가 433인 이유는 작은 코퍼스에서 목표 512개까지 merge를 만들지 못했기 때문이다. 정확한 hash와 special token은 `tokenizer-metadata.json`에 있다.

## 2. 같은 작은 모델의 DoRA SFT

위 40 update checkpoint를 base로 고정했다.

| 항목 | 실제 값 |
|---|---|
| 학습 가능 adapter 파라미터 | 20,736 |
| 전체 파라미터(base + adapter) | 147,136 |
| Method / rank / alpha / alpha÷rank | DoRA / 8 / 16 / 2 |
| 대상 projection | q, k, v, o, gate, up, down (각 층) |
| Train / validation | prompt-response 각각 1개 |
| Updates / batch / accumulation | 3 / 1 / 1 |
| Sequence length / precision | 128 / CUDA BF16 autocast |
| AdamW peak LR | 1e-4 |
| Betas / epsilon / decay | [0.9,0.95] / 1e-8 / 0.1; 1차원 magnitude는 decay 0 |
| Schedule | warmup 1 update + cosine; 종료 LR 1e-5 |
| Gradient clipping / Z-loss | 1.0 / 0 |
| Loss | 응답 token과 EOS만 계산, prompt 제외 |
| Seed | SFT CLI가 별도로 고정하지 않았음; 초기 adapter의 정확한 재현은 보장 안 됨 |

| 당시 결과 | 값 |
|---|---:|
| Train CE update 1 | 5.3890410203 |
| Train CE update 2 | 5.3804133489 |
| Train CE update 3 | 5.3739013672 |
| 각 update 유효 target | 13 |
| Validation CE / perplexity | 5.4805573357 / 239.9804197761 |
| Validation 유효 target | 9 |

SFT validation은 기본 모델 validation과 데이터·loss mask가 다르므로 두 perplexity를 직접 비교하지 않는다. adapter 재로딩 후 `--chat`, greedy, 최대 8 token으로 실행한 생성 결과도 제어 토큰(`<|assistant|>` 반복)과 공백 수준이었다. 이 adapter는 실제 교통 응답용 모델이 아니다.

## 3. 316M 기본 구조의 GPU 1-update 검사

| 항목 | 실제 값 |
|---|---|
| 모델 파라미터 | 315,936,768 |
| Layer / hidden / FFN | 24 / 1,024 / 2,816 |
| Q heads / KV heads / head dimension | 16 / 8 / 64 |
| Vocabulary / 모델 문맥 상한 | 32,000 / 4,096 |
| 실제 입력 | ID 1~32의 임의 token 나열, 한국어 텍스트 아님 |
| 실제 sequence length / batch / accumulation | 32 / 1 / 1 |
| Updates / 유효 target | 1 / 31 |
| Seed | 42 |
| Precision / 장치 | BF16 autocast / RTX 3060 |
| Optimizer | AdamW, LR 3e-4, betas [0.9,0.95], eps 1e-8 |
| Weight decay / clipping / Z-loss | 0.1(embedding·norm 제외) / 1.0 / 0 |
| Scheduler 설정 | total_steps=2, warmup_ratio=0.01, min_lr_ratio=0.1; 실제 실행은 1 update |
| Norm·RoPE·dropout·공유 head | 작은 모델과 동일 방식; RMS epsilon 1e-6, theta 10000, dropout 0 |
| Attention backend | SDPA |

결과: CE **10.5660420079**, clipping 전 gradient norm **30.1113586426**, 최대 CUDA allocated memory **5.8443717957 GiB**. 유한한 loss/gradient로 optimizer update가 완료됐다. CUDA reserved 메모리나 프로세스 전체 VRAM이 아니며 **4K 문맥 학습에 필요한 메모리로 해석하면 안 된다**. 이 모델의 가중치는 저장하지 않았다.

## 4. 자동 테스트에서 수행한 별도 업데이트

당시 `pytest` 18개가 통과했다. 이 중에는 작은 synthetic ID 모델을 30 update 과적합시키는 테스트, 중단/재개 비교, 누적 gradient 비교, LoRA/DoRA optimizer step이 포함된다. 따라서 테스트도 일부 실제 가중치 업데이트를 수행했다. 테스트 모델 기본값은 2층, hidden 32, Q/KV 4/2, head dimension 8, FFN 64, vocabulary 48, 최대 문맥 32다. 정확한 절차·seed·변형은 `tests/test_training.py`, `tests/test_adapters.py`, `tests/test_model.py`에 있다. 테스트들은 각자 초기화한 모델이며 위 세 실행의 연속 학습이 아니다.

이번 기록 정리에서는 이러한 테스트도 다시 실행하지 않았다. 학습을 금지한 상태에서 전체 pytest나 smoke를 실행하면 안 된다. 정적 문법 검사·JSON 확인·git diff 검사만 수행한다.

## 5. 기록 출처와 저장 범위

- `checkpoint-summary.json`: 저장된 base checkpoint의 config·step·cursor·optimizer·scheduler·run metadata를 읽기만 해서 추출.
- `model-config.json`, `tokenizer-metadata.json`, `adapter-manifest.json`: 실행 산출물의 설정 복사.
- `train.txt`, `validation.txt`, `sft-train.jsonl`, `sft-validation.jsonl`: 사용했던 짧은 예문 그대로 복사.
- CE·PPL·GPU 메모리·생성 결과: 당시 대화에 남은 콘솔 출력에서 전사. 원본 전체 로그는 별도 보관되지 않음.
- 모델/adapter 바이너리와 학습 tokenizer는 기존처럼 로컬 `checkpoints/smoke-implementation/`에 보존한다. Git에는 코드·논문 설계·설정·텍스트 결과 기록을 올린다.

학습 시간·tokens/sec·KOBEST 점수·실서비스 품질은 측정하지 않았으므로 추정값을 기록하지 않았다.
