# 첫 번째 한국어 LLM 설계

> 초기 MHA 설계 기록이다. 2026-09-17 이후의 구조 결정·설정은 [아키텍처 v1](architecture.md)을 기준으로 한다. 아래 약 50M 수치는 초기 목표 호칭이며 실측 파라미터 수가 아니다. 새 설계는 GQA와 headwise QK-Norm을 포함하고 40M/316M/1.2B로 구분한다.

이 문서는 `make-llm` 프로젝트에서 처음 완주할 기본 모델의 구조와 학습 원칙을 기록한다. 목표는 복잡한 최신 기법을 모두 넣는 것이 아니라, **작지만 제대로 학습되고 확장 가능한 한국어 Dense Decoder-only Transformer**를 만드는 것이다.

## 1. 한 줄 결론

첫 모델은 **Dense Decoder-only Transformer**로 만든다.

```text
텍스트
  → BPE Tokenizer
  → Token Embedding
  → Transformer Block × 8
  → RMSNorm
  → LM Head
  → 다음 Token 확률
```

- **Decoder-only**: 앞에 나온 token만 보고 다음 token을 예측한다. OLMo는 이 구조를 채택한 실제 공개 LLM이다. ([OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1))
- **Dense**: 각 token이 매 layer에서 모델의 모든 가중치를 사용한다. OLMo 2는 자신의 모델을 dense autoregressive language model로 설명한다. ([OLMo 2](https://arxiv.org/abs/2501.00656))
- 따라서 이 프로젝트에서 말하는 기본 모델은 `Dense Decoder-only Transformer`다.

## 2. 첫 모델 권장 설정

약 50M parameter를 목표로 하는 첫 설정이다. 작아서 오류를 찾고 반복 실험하기 좋으며, 이후 같은 구조를 더 크게 확장할 수 있다.

아래 표의 **프로젝트 결정**은 논문이 그 수치를 유일한 정답으로 제시했다는 뜻이 아니다. 이 저장소의 첫 완주 모델을 위한 출발점이다. 반면 나머지 선택은 링크한 원 논문에서 사용하거나 근거로 제시한 방식이다.

```yaml
model:
  vocab_size: 32000
  context_length: 1024
  hidden_size: 512
  num_layers: 8
  num_attention_heads: 8
  num_kv_heads: 8
  head_dim: 64
  ffn_type: swiglu
  intermediate_size: 1408
  positional_embedding: rope
  norm: rmsnorm
  tie_word_embeddings: true
  bias: false
  dropout: 0.0
```

| 항목 | 선택 | 이유 | 근거 논문 / 구분 |
| --- | --- | --- | --- |
| Tokenizer | BPE, 32K vocabulary | BPE는 subword 단위라 한국어·영어·코드 혼합을 한 tokenizer로 다룰 수 있다. 32K는 작은 첫 모델의 embedding 크기를 과도하게 키우지 않기 위한 시작값이다. | OLMo도 BPE 기반 tokenizer 사용: [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1). 32K와 SentencePiece 구현 선택: **프로젝트 결정**. |
| Context length | 1,024 | 초기 학습과 attention 메모리 부담을 낮춘다. 기본 학습이 끝나면 2,048로 늘려 비교한다. | OLMo는 2,048, OLMo 2는 4,096까지 확장: [OLMo 2 §2.1](https://arxiv.org/html/2501.00656v3#S2.SS1). 1,024 시작: **프로젝트 결정**. |
| Layer / hidden size | 8 / 512 | 작아서 전체 학습·생성·저장 흐름을 반복하기 좋다. `hidden_size=512`와 8 heads는 Transformer-base의 검증된 폭/heads 조합을 따른다. 8층은 첫 모델용 확장값이다. | 512 hidden, 8 heads: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 8 layers: **프로젝트 결정**. |
| Attention | MHA, 8 heads | 가장 구현·검증하기 단순하다. `512 / 8 = 64`이므로 head 하나의 크기는 64이다. | MHA의 기본 구조: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). OLMo 2도 7B·13B는 MHA, 32B에서만 GQA: [OLMo 2 Table 3](https://arxiv.org/html/2501.00656v3#S2.T3). |
| Position | RoPE | token 순서와 상대 거리를 attention에 전달한다. 절대 위치 embedding을 대체하는 방식이다. | [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1), [OLMo 2 §2.1](https://arxiv.org/html/2501.00656v3#S2.SS1). |
| Normalization | Pre-RMSNorm | 각 sublayer **입력**을 RMSNorm으로 정규화해 첫 구현의 안정적인 기본으로 삼는다. | 읽은 OLMo는 input 정규화, OLMo 2는 output RMSNorm + QK-Norm을 사용: [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1), [OLMo 2 §3.3](https://arxiv.org/html/2501.00656v3#S3.SS3). Pre-RMSNorm 자체는 **프로젝트 결정**. |
| FFN | SwiGLU, 1,408 | SwiGLU를 쓰고 내부 폭을 약 `8/3 × hidden_size`로 잡는다. `512 × 8/3 ≈ 1,365`를 GPU 친화적으로 128의 배수인 1,408로 올렸다. | SwiGLU와 `8/3d`, 128 배수 반올림: [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1). |
| Embedding / output | Weight tying | 입력 token embedding과 출력 LM Head 가중치를 공유해 parameter 수를 줄인다. | OLMo 1B가 weight tying을 사용: [OLMo Table 1](https://arxiv.org/html/2402.00838v4#S1.T1). 첫 50M 모델 적용: **프로젝트 결정**. |
| Bias / dropout | bias 없음, dropout 0.0 | bias는 제외해 구조를 단순하게 한다. dropout 0.0은 충분한 사전학습 데이터를 가정한 출발값이며, 작은 데이터에서는 0.05도 비교한다. | bias 없음: [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1). dropout 0.0: **프로젝트 결정**. |

## 3. Transformer block 내부

block 하나는 아래 순서로 작동한다.

```text
x
 ├─ RMSNorm → Causal Self-Attention(MHA + RoPE) → 더하기(Residual)
 └─ RMSNorm → SwiGLU FFN                       → 더하기(Residual)
```

| 구성 | 역할 | 읽은 논문 근거 |
| --- | --- | --- |
| Causal Self-Attention | 현재 token은 자기 자신과 과거 token만 참고한다. 미래 정답을 미리 보는 것을 막아 다음 token 예측이 가능하다. | Transformer decoder의 masked self-attention: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). OLMo는 이 decoder-only 구조를 채택: [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1). |
| MHA | 여러 관점(head)으로 token 사이의 관계를 본다. | [Attention Is All You Need](https://arxiv.org/abs/1706.03762), OLMo 2 7B·13B의 MHA: [OLMo 2 Table 3](https://arxiv.org/html/2501.00656v3#S2.T3). |
| RoPE | token의 위치와 상대적 거리를 Q/K에 반영한다. | [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1), [OLMo 2 §2.1](https://arxiv.org/html/2501.00656v3#S2.SS1). |
| RMSNorm | 값의 크기를 안정화한다. | OLMo 2가 RMSNorm으로 전환: [OLMo 2 §3.3](https://arxiv.org/html/2501.00656v3#S3.SS3). |
| Residual connection | block 입력을 결과에 더해 정보와 gradient가 깊은 layer까지 전달되게 한다. | Transformer의 Add & Norm residual 구조: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). |
| SwiGLU FFN | attention 후 token 내부의 정보를 넓혔다가 다시 줄이며 가공한다. | OLMo가 ReLU 대신 SwiGLU 사용: [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1). |

## 4. 지금은 넣지 않는 것

| 항목 | 지금 제외하는 이유 | 나중에 넣을 시점 | 읽은 논문 근거 / 구분 |
| --- | --- | --- | --- |
| GQA | KV cache를 줄이지만 첫 구현에는 MHA가 더 단순하다. OLMo 2도 7B·13B에는 MHA를 사용했다. | 300M 이상 또는 긴 context 추론에서 메모리가 문제일 때. | [OLMo 2 Table 3](https://arxiv.org/html/2501.00656v3#S2.T3): 32B에서 GQA로 전환. 300M 기준은 **프로젝트 결정**. |
| MoE | routing, load balancing, 분산 학습까지 필요해 복잡하다. | Dense 모델을 안정적으로 학습한 뒤 | 읽은 논문에 MoE 구현 근거는 없음. **프로젝트 범위 결정**. |
| QK-Norm / Z-Loss | 학습 안정화에 도움이 되지만, 기본 구조의 오류를 분리하기 어렵게 만든다. | 기본 모델의 loss가 정상적으로 감소한 뒤 안정화 실험으로 | OLMo 2는 QK-Norm과 Z-Loss로 안정성을 개선: [OLMo 2 §3.3](https://arxiv.org/html/2501.00656v3#S3.SS3). 적용 시점은 **프로젝트 결정**. |
| FlashAttention 등 특수 kernel | 빠르지만 알고리즘 구현과 성능 최적화를 섞게 된다. | attention 정합성 테스트 후 성능 개선 단계에서 | 읽은 논문의 핵심 주제는 아님. **프로젝트 구현 순서 결정**. |
| LoRA / QLoRA | base model을 처음부터 학습하는 기술이 아니라, 완성된 모델을 적은 비용으로 특화하는 방법이다. | base 모델 완성 후 DAPT/SFT 단계에서 | [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314). DAPT의 필요성: [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964). |

## 5. 학습 기본값

```yaml
training:
  precision: bf16
  optimizer: AdamW
  learning_rate: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  scheduler: cosine_decay
  warmup_ratio: 0.01
  gradient_clip_norm: 1.0
```

| 항목 | 선택 이유 | 읽은 논문 근거 / 구분 |
| --- | --- | --- |
| BF16 | 메모리를 아끼면서 학습에 적합한 수치 형식이다. | OLMo는 mixed precision에서 bfloat16을 사용: [OLMo §3.1](https://arxiv.org/html/2402.00838v4#S3.SS1). |
| AdamW, betas `[0.9, 0.95]` | LLM pretraining의 출발 optimizer로 삼는다. | OLMo의 모든 run이 이 AdamW beta를 사용: [OLMo Table 1](https://arxiv.org/html/2402.00838v4#S1.T1). |
| Learning rate `3e-4` | 첫 실험의 peak learning rate다. | OLMo 7B와 OLMo 2 7B가 `3e-4`를 사용: [OLMo Table 1](https://arxiv.org/html/2402.00838v4#S1.T1), [OLMo 2 Table 3](https://arxiv.org/html/2501.00656v3#S2.T3). 50M에 그대로 최적인지는 미검증이므로 **프로젝트 시작값**. |
| Weight decay `0.1` | weight가 과도하게 커지는 것을 억제한다. | [OLMo §3.2](https://arxiv.org/html/2402.00838v4#S3.SS2). |
| Warmup + cosine decay | 초반에는 learning rate를 올리고, 후반에는 부드럽게 내린다. | OLMo 2가 warmup 후 cosine decay를 사용: [OLMo 2 §2.3](https://arxiv.org/html/2501.00656v3#S2.SS3). `warmup_ratio=0.01`은 **프로젝트 시작값**. |
| Gradient clipping `1.0` | gradient가 비정상적으로 커져 학습이 무너지는 것을 막는다. | OLMo의 clip norm 1.0: [OLMo §3.2](https://arxiv.org/html/2402.00838v4#S3.SS2). |

위 값은 출발점이다. 고정된 정답이 아니므로 loss, gradient norm, validation perplexity, 생성 예시를 기록하면서 조정한다.

## 6. 데이터와 확장 순서

모델 구조만큼 데이터 품질과 양이 중요하다. OLMo는 언어·품질·내용 필터링, 중복 제거, 여러 데이터 출처 혼합을 거쳐 데이터를 만들었다. ([OLMo §2.2](https://arxiv.org/html/2402.00838v4#S2.SS2)) 50M 모델의 약 10억 token 목표는 이 프로젝트의 예산·실험용 시작점이며, 읽은 논문이 50M에 대해 보장한 값은 아니다.

| 순서 | 할 일 | 읽은 논문 근거 / 구분 |
| --- | --- | --- |
| 1 | Character tokenizer + 아주 작은 모델로 전체 학습 흐름 검증 | **프로젝트 구현 순서 결정**. |
| 2 | BPE tokenizer + 위의 50M 기본 모델 완성 | BPE 기반 tokenizer: [OLMo §2.1](https://arxiv.org/html/2402.00838v4#S2.SS1). 50M 크기: **프로젝트 결정**. |
| 3 | 768 hidden / 16 layers 규모로 확장 | **프로젝트 확장 단계 결정**. |
| 4 | 1024 hidden / 24 layers 규모로 확장 | **프로젝트 확장 단계 결정**. |
| 5 | 고품질 분야 데이터로 DAPT | 분야 데이터로 추가 사전학습: [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964). |
| 6 | 지시 데이터로 SFT, 필요하면 LoRA 또는 QLoRA | LoRA/QLoRA의 효율적인 fine-tuning: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314). |

## 7. 참고한 논문

- [Attention Is All You Need](papers/README.md#1-attention-is-all-you-need--transformer): Transformer의 기본 원리
- [OLMo / OLMo 2](papers/README.md#5-olmo): 실제 decoder-only LLM의 구조와 학습 레시피
- [Don't Stop Pretraining](papers/README.md#2-dont-stop-pretraining): 분야 특화 추가 학습(DAPT/TAPT)
- [LoRA / QLoRA](papers/README.md#3-lora): 완성된 base 모델의 효율적인 fine-tuning
