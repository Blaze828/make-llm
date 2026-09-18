# model

현재 기준: [아키텍처 v1](../docs/architecture.md). `config.py`, `cache.py`, `norm.py`, `rope.py`, `attention.py`, `mlp.py`, `transformer.py`를 구현했다. 출력 dataclass는 별도 `outputs.py` 대신 `transformer.py`에 둔다. [실행·검증 안내](../docs/implementation.md).

| 파일 | 계약 |
|---|---|
| `config.py` | JSON 읽기; hidden_size=Q heads×head_dim, Q heads%KV heads=0, 짝수 head_dim, 양수 차원 검증 |
| `cache.py` | 층별 회전된 K·원래 V, 절대 위치·문맥 상한, 요청별 cache 소유권 관리 |
| `transformer.py`의 `ModelOutput` | logits와 선택적 past_key_values 반환 형식 |

`attention.py`는 MHA를 KV heads=Q heads인 특수 경우로 지원한다. `norm.py`는 Pre/Final RMSNorm과 headwise QK-RMSNorm을 제공한다. `transformer.py`는 embedding/head 공유를 보장한다. loss shift·sampling·optimizer는 모델 밖에서 담당한다.

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
