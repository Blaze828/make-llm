# 읽은 논문 정리

`make-llm`을 설계할 때 참고한 핵심 LLM 논문과 관련 용어를 모은 노트다. 논문 원문을 다시 볼 때는 각 논문의 **핵심 질문**, **핵심 방법**, **한 문장 요약**부터 확인한다.

## 한눈에 보기

| 논문 | 프로젝트에서 답하는 질문 | 핵심 아이디어 |
| --- | --- | --- |
| [Attention Is All You Need](#1-attention-is-all-you-need--transformer) | RNN/CNN 없이 문장을 처리할 수 있을까? | Attention만 사용하는 Transformer |
| [Don't Stop Pretraining](#2-dont-stop-pretraining) | 범용 모델을 특정 분야·작업에 맞출 수 있을까? | DAPT와 TAPT |
| [LoRA](#3-lora) | 거대한 LLM 전체를 재학습하지 않을 수 있을까? | 작은 low-rank 행렬만 학습 |
| [QLoRA](#4-qlora) | LoRA의 GPU 메모리도 더 줄일 수 있을까? | 4-bit base model 위의 LoRA 학습 |
| [OLMo](#5-olmo) | LLM을 처음부터 실제로 어떻게 만들고 학습할까? | 공개된 end-to-end 학습 레시피 |
| [OLMo 2](#6-olmo-2) | 더 안정적이고 효율적으로 학습하려면? | 안정화, 고품질 mid-training, post-training |
| [HyperCLOVA](#7-hyperclova) | 한국어에 맞는 대규모 생성형 LLM은 어떻게 만들까? | 한국어 데이터와 Morpheme-aware Byte-level BPE |
| [Polyglot-Ko](#8-polyglot-ko) | 공개 한국어 LLM의 데이터·구조·평가는 어떻게 구성할까? | GPT-NeoX, 한국어 전처리, RoPE, 형태소 인지 BPE |
| [DoRA](#9-dora) | LoRA를 더 표현력 있게 만들 수 있을까? | Weight의 magnitude와 direction을 분리해 학습 |

## 1. Attention Is All You Need — Transformer

원문: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | RNN/CNN 없이 문장을 처리할 수 있을까? |
| 핵심 방법 | Attention만 사용하는 Transformer 제안 |
| 가장 중요한 특징 | 병렬 처리 가능, 긴 거리 관계 학습에 유리 |
| 구조 | Encoder 6층 + Decoder 6층 |
| Hidden size / FFN size | 512 / 2,048 |
| Attention heads | 8 |
| 모델 크기 | Base 약 65M, Big 약 213M |
| Optimizer | Adam |
| 핵심 용어 | Self-Attention, Multi-Head Attention, Q/K/V, FFN, Positional Encoding |

**한 문장 요약:** 현재 대부분의 LLM의 기본 구조가 된 Transformer를 처음 제안한 논문이다.

## 2. Don't Stop Pretraining

원문: [arXiv:2004.10964](https://arxiv.org/abs/2004.10964)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | 범용 언어 모델을 특정 분야에 더 잘 맞출 수 있을까? |
| 핵심 방법 | DAPT + TAPT |
| Base model | RoBERTa-base |
| DAPT | 특정 domain 데이터로 추가 사전학습 |
| TAPT | 특정 task 데이터로 추가 사전학습 |
| 학습 방식 | Masked Language Modeling |
| 핵심 발견 | 데이터 양보다 domain/task와의 관련성이 중요 |
| 좋은 흐름 | Pretraining → DAPT → TAPT → Fine-tuning |

**한 문장 요약:** 범용 모델도 실제 사용할 분야와 task의 데이터로 계속 학습시키면 더 좋아진다.

## 3. LoRA

원문: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | 거대한 LLM 전체를 다시 학습하지 않고 fine-tuning할 수 있을까? |
| 핵심 방법 | 기존 weight는 고정하고 작은 low-rank 행렬 `A`, `B`만 학습 |
| 핵심 수식 | $W = W_0 + BA$ |
| Base weight / 학습 대상 | Freeze / `A`, `B` |
| 핵심 hyperparameter | Rank `r` |
| 대표 실험 | GPT-3 175B |
| 가장 중요한 발견 | Rank를 무조건 키우는 것보다 어느 weight에 LoRA를 적용하는지가 중요 |
| 대표 적용 위치 | `Wq`, `Wv` |

**한 문장 요약:** 전체 모델 대신 아주 작은 추가 파라미터만 학습해서 fine-tuning 비용을 크게 줄이는 방법이다.

## 4. QLoRA

원문: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | LoRA도 base model 때문에 GPU 메모리가 큰데 더 줄일 수 없을까? |
| 핵심 방법 | 4-bit base model + LoRA |
| Base model | 4-bit로 quantization 후 freeze |
| LoRA | BF16 등으로 학습 |
| 핵심 기술 | NF4, Double Quantization, Paged Optimizer |
| 대표 결과 | 65B 모델을 단일 48GB GPU에서 fine-tuning |
| LoRA rank | 대표적으로 64 |
| 중요한 발견 | Rank보다 모든 linear layer에 LoRA를 적용하는 것이 중요 |
| 데이터 관련 발견 | 데이터 양보다 데이터 품질이 중요 |

**한 문장 요약:** Base LLM을 4-bit로 압축하고 LoRA만 학습해서 매우 적은 GPU 메모리로 fine-tuning하는 방법이다.

## 5. OLMo

원문: [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | 실제 LLM을 처음부터 어떻게 만들고 학습할까? |
| 핵심 특징 | 데이터, 코드, weight, checkpoint, 로그까지 공개 |
| Architecture | Decoder-only Transformer |
| 대표 모델 / layers | OLMo-7B / 32 |
| Hidden size / attention heads | 4,096 / 32 |
| FFN/SwiGLU size | 11,008 |
| Context length | 2,048 |
| Tokenizer | BPE |
| Training tokens | 약 2.46T |
| Optimizer / peak learning rate | AdamW / 3e-4 |
| 핵심 구조 | RoPE, SwiGLU, no bias, LayerNorm |
| 핵심 발견 | 학습 token 양뿐 아니라 training data의 분포와 품질도 중요 |

**한 문장 요약:** 실제 7B급 LLM을 처음부터 학습하는 전체 과정을 공개한 논문이다.

## 6. OLMo 2

원문: [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | OLMo를 더 안정적이고 효율적으로 학습하려면? |
| 대표 모델 | 7B / 13B / 32B |
| Architecture | Decoder-only Transformer |
| 7B layers / hidden size | 32 / 4,096 |
| 13B layers / hidden size | 40 / 5,120 |
| 32B layers / hidden size | 64 / 5,120 |
| Attention | 7B·13B = MHA, 32B = GQA |
| Context length | 4,096 |
| 핵심 안정화 | RMSNorm, QK-Norm, Z-Loss, 데이터 필터링 |
| 학습 단계 | Pretraining → Mid-training → SFT → DPO → RLVR |
| 가장 큰 성능 향상 | 고품질 mid-training 데이터 |
| 대표 token 수 | 최대 약 6.6T |

**한 문장 요약:** 모델 크기뿐 아니라 학습 안정성, 고품질 mid-training, post-training이 최종 성능에 매우 중요하다는 것을 보여준다.

## 7. HyperCLOVA

원문: [What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers](https://aclanthology.org/2021.emnlp-main.274.pdf)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | 한국어에 특화된 대규모 생성형 LLM을 어떻게 만들 수 있을까? |
| 구조 | GPT-3 계열의 autoregressive Decoder-only Transformer |
| 학습 데이터 | 블로그, 뉴스, 카페, 댓글, Q&A 등 다양한 한국어 중심 데이터 |
| Tokenizer | Morpheme-aware Byte-level BPE |
| Tokenizer 핵심 발견 | 일반 Byte-level BPE 및 Character BPE보다 한국어 task에서 더 높은 성능을 보임 |
| 성능 경향 | 모델 크기가 커질수록 few-shot 성능이 전반적으로 향상 |

### 프로젝트에서 참고할 점

한국어 LLM의 성능은 모델 크기 하나로 결정되지 않는다.

```text
한국어 데이터 품질
  + 데이터 다양성
  + 한국어에 적합한 Tokenizer
  + 모델 크기
```

이 프로젝트에서는 HyperCLOVA를 **한국어 데이터 구성과 tokenizer 설계의 참고 모델**로 사용한다. 특히 단순 BPE부터 구현한 다음, 한국어 형태소 특성을 반영한 Morpheme-aware Byte-level BPE를 비교 실험 후보로 둔다.

**한 문장 요약:** 한국어 중심의 다양한 데이터와 한국어에 맞춘 tokenizer가 대규모 한국어 LLM의 핵심 요소임을 보여준다.

## 8. Polyglot-Ko

원문: [A Technical Report for Polyglot-Ko: Open-Source Large-Scale Korean Language Models](https://arxiv.org/pdf/2306.02254)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | 공개 한국어 LLM의 데이터, 구조, 평가는 어떻게 구성할까? |
| 기반 구조 | EleutherAI GPT-NeoX 기반 Decoder-only Transformer |
| 모델 크기 | 1.3B / 3.8B / 5.8B / 12.8B |
| Raw data | 약 1.2TB의 한국어 중심 원천 데이터 수집 |
| 데이터 종류 | 블로그, 뉴스, Q&A, 특허, 소설, 댓글 등 |
| 전처리 | 중복 제거, 개인정보 제거, HTML 정리 |
| Tokenizer | MeCab + Morpheme-aware Byte-level BPE |
| 위치 정보 | RoPE (Rotary Positional Embedding) |
| 평가 | KOBEST로 zero-shot / few-shot 평가 |

### 프로젝트에서 참고할 점

한국어 Base LLM의 데이터 흐름은 다음과 같이 설계한다.

```text
한국어 Raw Data
        ↓
Cleaning / Deduplication / PII 제거
        ↓
Tokenizer
        ↓
Token Embedding
        ↓
Decoder Transformer × N
 ├─ Causal Self-Attention
 ├─ RoPE
 └─ MLP
        ↓
Linear Layer
        ↓
Next Token Prediction
```

모델 평가는 loss만으로 끝내지 않는다.

```text
Training Loss
  + Validation Loss
  + 실제 Text Generation
  + 한국어 Benchmark
```

**한 문장 요약:** 한국어 LLM에서는 GPT-NeoX 계열 Decoder-only 구조뿐 아니라 데이터 정제, 형태소 인지 tokenizer, 한국어 benchmark 평가가 함께 필요하다.

## 9. DoRA

원문: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/pdf/2402.09353)

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | 적은 parameter만 학습하면서 LoRA보다 full fine-tuning에 가까운 업데이트를 만들 수 있을까? |
| 종류 | PEFT (Parameter-Efficient Fine-Tuning) |
| 핵심 방법 | pre-trained weight를 magnitude와 direction으로 분리 |
| Magnitude | 별도 parameter로 직접 학습 |
| Direction | LoRA의 low-rank update로 학습 |
| 결과 | 비슷한 학습 parameter 수에서 여러 LLM·멀티모달 실험의 LoRA보다 높은 성능 보고 |
| 확장 | 4-bit quantization과 결합한 QDoRA 사용 가능 |

LoRA는 기존 weight에 low-rank update를 더한다.

```text
W' = W₀ + BA

W₀: 기존 pre-trained weight
A, B: 학습하는 low-rank matrix
```

DoRA는 weight를 크기와 방향으로 나눈다.

```text
Weight
 ├─ Magnitude (크기)  → 직접 학습
 └─ Direction (방향)  → LoRA 방식으로 학습
```

개념적으로 DoRA의 업데이트는 다음처럼 표현할 수 있다.

```text
W' = m × (W₀ + BA) / ||W₀ + BA||

m: 학습하는 magnitude
W₀ + BA: LoRA로 갱신한 direction
```

### LoRA와 DoRA 비교

| 방식 | 업데이트 방법 | 프로젝트에서의 역할 |
| --- | --- | --- |
| LoRA | low-rank update가 magnitude와 direction 변화를 함께 표현 | 가장 먼저 구현할 PEFT 기준선 |
| DoRA | magnitude는 직접, direction은 low-rank update로 분리 | LoRA 기준선 뒤에 비교할 고성능 PEFT 후보 |
| QLoRA | 4-bit base model + LoRA | GPU 메모리가 부족할 때의 LoRA 방식 |
| QDoRA | 4-bit base model + DoRA | GPU 메모리가 부족할 때의 DoRA 실험 후보 |

**한 문장 요약:** DoRA는 LoRA의 간결함을 유지하면서 weight의 크기와 방향을 분리해 더 표현력 있는 fine-tuning을 목표로 한다.

## 10. 논문의 관계와 프로젝트 적용 방향

```text
Transformer
  └─ LLM의 기본 구조를 만듦

HyperCLOVA / Polyglot-Ko
  └─ 한국어 데이터, 형태소 인지 tokenizer, Decoder-only 구조의 참고

OLMo / OLMo 2
  └─ 실제 LLM을 처음부터 pretraining

Don't Stop Pretraining
  └─ 특정 domain / task에 추가 적응

LoRA
  └─ 적은 parameter만 fine-tuning

QLoRA
  └─ base model까지 4-bit로 줄여 더 적은 GPU로 fine-tuning

DoRA
  └─ LoRA의 magnitude / direction 분리 방식으로 fine-tuning
```

프로젝트는 HyperCLOVA와 Polyglot-Ko를 참고해 한국어 Base LLM을 만들고, DoRA로 전체 모델을 다시 학습하지 않고 새 분야에 적응하는 것을 목표로 한다.

```text
한국어 데이터 수집 및 정제
          ↓
한국어 Tokenizer 구축
          ↓
Decoder-only Transformer 구현
          ↓
Next Token Prediction 기반 Pre-training
          ↓
한국어 Base LLM
          ↓
DoRA 기반 Fine-Tuning
          ↓
특정 Domain에 특화된 LLM
```

## 11. 기본 모델 구조 용어


| 용어 | 의미 |
| --- | --- |
| Transformer | Attention을 중심으로 만든 신경망 구조 |
| Decoder-only | Encoder 없이 decoder 구조만 사용하는 LLM 방식 |
| Layer | Transformer block 하나. 여러 층을 쌓아 모델을 만듦 |
| Hidden size / `d_model` | Token 하나를 내부에서 몇 차원 벡터로 표현하는지 |
| Parameter | 모델이 학습하면서 바꾸는 숫자 |
| Context length | 모델이 한 번에 처리하도록 학습되는 token 길이 |
| Embedding | Token을 숫자 벡터로 바꾸는 과정 |
| Vocabulary | Tokenizer가 사용할 수 있는 token 전체 목록 |

예를 들어 hidden size가 4,096이면 `사과`라는 token 하나가 내부에서 4,096개의 숫자로 표현된다는 뜻이다.

## 12. Attention 관련 용어

| 용어 | 의미 |
| --- | --- |
| Attention | 어떤 token이 다른 token을 얼마나 참고할지 계산 |
| Self-Attention | 같은 문장 안의 token끼리 서로 attention |
| Multi-Head Attention (MHA) | 여러 attention head를 동시에 사용 |
| Head | 하나의 attention 계산 단위 |
| Q (Query) | 내가 어떤 정보를 찾고 있는가 |
| K (Key) | 내가 어떤 정보를 가지고 있는가 |
| V (Value) | 실제로 전달할 정보 |
| GQA | 여러 query head가 K/V를 공유하는 attention |
| QK-Norm | Query와 Key를 normalize해서 attention을 안정화 |

```text
Query + Key → 어떤 token이 중요한지 계산 → Value에서 정보 가져오기
```

## 13. FFN, Activation, 위치 정보

FFN(Feed Forward Network)은 attention 이후 각 token의 정보를 다시 가공하는 부분이다. Transformer Base에서는 보통 차원을 넓혔다가 다시 줄인다.

```text
512 → 2,048 → 512

Attention = token끼리 정보 교환
FFN       = 받은 정보를 각 token 내부에서 가공
```

SwiGLU는 현대 LLM의 FFN에서 많이 쓰는 activation 구조다. 초기 Transformer의 ReLU보다 최근 모델에서 더 자주 사용하며 OLMo와 OLMo 2에도 쓰인다.

Transformer는 구조만으로 token 순서를 알 수 없다. 따라서 `나는 너를 사랑해`와 `너를 나는 사랑해`처럼 순서가 다른 문장을 구분할 위치 정보가 필요하다.

| 용어 | 의미 |
| --- | --- |
| Positional Encoding | Token의 순서를 알려주는 정보 |
| RoPE | Rotary Positional Embedding. 현대 LLM에서 많이 사용하는 위치 표현 |

## 14. Normalization과 학습 안정화

| 용어 | 의미 |
| --- | --- |
| LayerNorm | 내부 값의 크기를 안정적으로 조절 |
| RMSNorm | LayerNorm을 단순화한 방식 |
| QK-Norm | Query/Key를 normalize |
| Z-Loss | Logit이 지나치게 커지는 것을 방지 |
| Gradient Clipping | Gradient가 너무 커지면 제한 |
| Loss Spike | 학습 중 loss가 갑자기 크게 튀는 현상 |

공통 목적은 **학습이 중간에 불안정해지거나 실패하지 않게 하는 것**이다.

## 15. Tokenizer

| 용어 | 의미 |
| --- | --- |
| Token | LLM이 실제로 처리하는 문자/단어 조각 |
| Tokenizer | 문장을 token으로 나누는 프로그램 |
| BPE | 자주 등장하는 문자 조각을 합쳐 token을 만드는 방식 |
| Vocabulary size | Token 종류의 수 |

예를 들어 `unbelievable`은 tokenizer에 따라 `un` → `believ` → `able`처럼 나뉠 수 있다.

## 16. 학습 단계

| 단계 | 의미 |
| --- | --- |
| Pretraining | 대규모 데이터로 언어와 일반 지식을 학습 |
| Mid-training | Pretraining 후반에 고품질·특정 능력 데이터를 집중 학습 |
| DAPT | 특정 domain 데이터로 추가 pretraining |
| TAPT | 특정 task 데이터로 추가 pretraining |
| Fine-tuning | 특정 목적에 맞게 모델을 추가 학습 |
| SFT | 질문-좋은 답변 데이터로 fine-tuning |
| DPO | 두 답변 중 어느 답이 좋은지 학습 |
| RLVR | 정답을 자동 검증할 수 있는 문제로 강화학습 |

```text
Pretraining → Mid-training → DAPT / TAPT → SFT → DPO / RLVR
```

모든 모델이 모든 단계를 거치는 것은 아니며, 목적에 따라 선택한다.

### DAPT와 TAPT의 차이

```text
DAPT: 일반 LLM → 의학 데이터 → 의학 분야에 적응
TAPT: 의학 모델 → 질병 분류 task 데이터 → 질병 분류에 더 특화
```

즉, **DAPT는 분야(domain)**, **TAPT는 작업(task)** 에 맞춘 추가 사전학습이다.

## 17. LoRA와 Rank

LoRA의 기본 수식은 $W = W_0 + BA$다.

| 용어 | 의미 |
| --- | --- |
| `W₀` | 기존 모델 weight. 학습하지 않고 고정(freeze) |
| `A`, `B` | 새롭게 추가하고 학습하는 작은 행렬 |
| `r` | LoRA rank |
| `α` | LoRA 변화량을 조절하는 scale |
| Freeze | Parameter를 학습하지 않고 고정 |

원래 12,288차원 공간의 변화량을 `12,288 → r=4 → 12,288`처럼 작은 차원을 통과시켜 표현할 수 있다. `r`이 작을수록 학습 parameter와 memory는 줄지만, rank를 계속 키운다고 성능이 계속 좋아지는 것은 아니다.

## 18. Quantization과 Precision

Quantization은 weight를 더 적은 bit로 저장하는 방법이다.

```text
FP32 → FP16 / BF16 → INT8 → 4-bit
```

bit 수가 줄수록 memory는 줄지만 정보 손실 가능성은 커진다.

| 용어 | 의미 |
| --- | --- |
| FP32 | 32-bit 부동소수점 |
| FP16 | 16-bit 부동소수점 |
| BF16 | 16-bit 형식. LLM 학습에서 많이 사용 |
| INT8 | 8-bit 정수 |
| NF4 | 신경망 weight 분포를 고려해 만든 4-bit 자료형 |
| Double Quantization | Quantization scale도 다시 quantize해 memory를 절약 |
| Paged Optimizer | GPU memory 부족 시 일부 optimizer 상태를 CPU RAM으로 옮겨 OOM을 줄이는 방법 |

## 19. Optimizer, Learning Rate, Warmup

Optimizer는 weight를 어떤 방식으로 업데이트할지 결정한다. 대표적으로 Adam과 AdamW가 있고 최근 LLM에는 AdamW가 많이 쓰인다.

Learning rate는 한 번 학습할 때 weight를 얼마나 크게 수정할지 결정한다. 너무 크면 loss spike와 학습 불안정이 생기고, 너무 작으면 학습이 매우 느려진다.

- **Warmup:** 처음에는 작은 learning rate로 시작해 점차 peak learning rate까지 올려 학습 초반을 안정화한다.
- **Learning-rate decay:** 학습 후반에는 learning rate를 점차 낮춰 weight를 세밀하게 조정한다.
- **Weight decay:** weight가 지나치게 커지는 것을 막는 regularization으로, L2 regularization과 관련이 있다.

---

읽은 논문이 늘어날 때 이 형식으로 계속 확장한다.
