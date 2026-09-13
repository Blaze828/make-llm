# 첫 번째 한국어 LLM 설계

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

- **Decoder-only**: 앞에 나온 token만 보고 다음 token을 예측한다. GPT, Llama, OLMo와 같은 방식이다.
- **Dense**: 각 token이 매 layer에서 모델의 모든 가중치를 사용한다. MoE처럼 일부 전문가만 고르는 구조가 아니다.
- 따라서 이 프로젝트에서 말하는 기본 모델은 `Dense Decoder-only Transformer`다.

## 2. 첫 모델 권장 설정

약 50M parameter를 목표로 하는 첫 설정이다. 작아서 오류를 찾고 반복 실험하기 좋으며, 이후 같은 구조를 더 크게 확장할 수 있다.

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

| 항목 | 선택 | 이유 |
| --- | --- | --- |
| Tokenizer | SentencePiece BPE, 32K vocabulary | 한국어·영어·코드가 섞인 데이터에도 무난하다. |
| Context length | 1,024 | 초기 학습과 디버깅의 GPU 부담을 줄인다. 안정화 후 2,048로 늘린다. |
| Layer / hidden size | 8 / 512 | 첫 완주 모델에 적당한 크기다. |
| Attention | MHA, 8 heads | 가장 단순하고 검증하기 쉽다. `512 / 8 = 64`이므로 head 하나의 크기는 64이다. |
| Position | RoPE | token 순서와 거리를 attention에 전달하는 현대적인 기본 방식이다. |
| Normalization | Pre-RMSNorm | 각 sublayer 앞에서 값을 안정화해 학습을 돕는다. |
| FFN | SwiGLU, 1,408 | attention으로 섞인 정보를 token별로 가공한다. ReLU FFN보다 권장되는 기본값이다. |
| Embedding / output | Weight tying | 입력 embedding과 출력 LM Head 가중치를 공유해 parameter를 줄인다. |
| Bias / dropout | bias 없음, dropout 0.0 | 대규모 언어모델의 단순한 기본 구성이다. 데이터가 매우 작을 때만 dropout 0.05를 실험한다. |

## 3. Transformer block 내부

block 하나는 아래 순서로 작동한다.

```text
x
 ├─ RMSNorm → Causal Self-Attention(MHA + RoPE) → 더하기(Residual)
 └─ RMSNorm → SwiGLU FFN                       → 더하기(Residual)
```

- **Causal Self-Attention**: 현재 token은 자기 자신과 과거 token만 참고한다. 미래 정답을 미리 보는 것을 막기 때문에 다음 token 예측이 가능하다.
- **MHA (Multi-Head Attention)**: 여러 관점(head)으로 token 사이의 관계를 본다.
- **RoPE**: token의 위치와 상대적 거리를 Q/K에 반영한다.
- **RMSNorm**: 값의 크기를 안정화한다.
- **Residual connection**: block 입력을 결과에 더해 정보와 gradient가 깊은 layer까지 잘 전달되게 한다.
- **SwiGLU FFN**: attention 후 token 내부의 정보를 넓혔다가 다시 줄이며 가공한다.

## 4. 지금은 넣지 않는 것

| 항목 | 지금 제외하는 이유 | 나중에 넣을 시점 |
| --- | --- | --- |
| GQA | KV cache를 줄이지만 첫 구현에는 MHA가 더 단순하다. | 300M 이상 또는 긴 context 추론에서 메모리가 문제일 때 |
| MoE | routing, load balancing, 분산 학습까지 필요해 복잡하다. | Dense 모델을 안정적으로 학습한 뒤 |
| QK-Norm / Z-Loss | 학습 안정화에 도움이 될 수 있으나, 기본 구조의 오류를 분리하기 어렵게 만든다. | 기본 모델의 loss가 정상적으로 감소한 뒤 안정화 실험으로 |
| FlashAttention 등 특수 kernel | 빠르지만 알고리즘 구현과 성능 최적화를 섞게 된다. | attention 정합성 테스트 후 성능 개선 단계에서 |
| LoRA / QLoRA | base model을 처음부터 학습하는 기술이 아니라, 완성된 모델을 적은 비용으로 특화하는 방법이다. | base 모델 완성 후 DAPT/SFT 단계에서 |

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

- **BF16**: 메모리를 아끼면서 학습에 적합한 수치 형식이다.
- **AdamW**: LLM 학습에서 널리 쓰는 optimizer다.
- **Warmup**: 초반 learning rate를 천천히 올려 불안정을 줄인다.
- **Cosine decay**: 후반으로 갈수록 learning rate를 부드럽게 줄인다.
- **Gradient clipping**: gradient가 비정상적으로 커져 학습이 무너지는 것을 막는다.

위 값은 출발점이다. 고정된 정답이 아니므로 loss, gradient norm, validation perplexity, 생성 예시를 기록하면서 조정한다.

## 6. 데이터와 확장 순서

모델 구조만큼 데이터 품질과 양이 중요하다. 50M 모델은 우선 정제·중복 제거한 데이터 약 10억 token을 목표로 하며, 부족하면 모델을 무리하게 키우지 않는다.

```text
1. Character tokenizer + 아주 작은 모델로 전체 학습 흐름 검증
2. BPE tokenizer + 위의 50M 기본 모델 완성
3. 768 hidden / 16 layers 규모로 확장
4. 1024 hidden / 24 layers 규모로 확장
5. 고품질 분야 데이터로 DAPT
6. 지시 데이터로 SFT, 필요하면 LoRA 또는 QLoRA
```

## 7. 참고한 논문

- [Attention Is All You Need](papers/README.md#1-attention-is-all-you-need--transformer): Transformer의 기본 원리
- [OLMo / OLMo 2](papers/README.md#5-olmo): 실제 decoder-only LLM의 구조와 학습 레시피
- [Don't Stop Pretraining](papers/README.md#2-dont-stop-pretraining): 분야 특화 추가 학습(DAPT/TAPT)
- [LoRA / QLoRA](papers/README.md#3-lora): 완성된 base 모델의 효율적인 fine-tuning
