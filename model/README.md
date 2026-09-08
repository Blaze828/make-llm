# model

Base LLM 구조. (README 7번)

| 파일 | 역할 |
|---|---|
| `norm.py` | RMSNorm |
| `rope.py` | 회전 위치 인코딩 |
| `attention.py` | Causal Self-Attention (GQA) |
| `mlp.py` | SwiGLU |
| `transformer.py` | Block 조립 + GPT 본체 (Embedding → Block×N → LM Head) |

검증된 구조를 그대로 따른다. 새로운 구조 실험은 Base가 정상 학습된 뒤에.

어댑터(LoRA 등) 코드는 여기 넣지 않는다. `adapters/`로 분리한다. (README 8번)
