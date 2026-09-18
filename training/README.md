# training

사전학습과 재학습 루프.

| 파일 | 역할 |
|---|---|
| `dataloader.py` | input_ids·문서/패딩 mask 제공. loss 모듈이 정답 shift를 한 번만 수행 |
| `loss.py` | shifted causal CE와 선택적 Z-loss, 유효 target만 FP32 계산 |
| `optimizer.py` | AdamW, 공유 weight 중복 제외·embedding/norm no-decay 그룹 |
| `pretrain.py` | Base LLM 사전학습 |
| `finetune.py` | 응답 전용 SFT (LoRA / DoRA). Full FT·QLoRA CLI는 후속 범위 |

하이퍼파라미터는 코드에 박지 않고 `configs/`에서 읽는다. (README 18번)

위 파일을 구현했다. 추가로 `engine.py`, `checkpoint.py`, `generate.py`가 있다. [실행 안내](../docs/implementation.md)를 따른다. AdamW·cosine·gradient clipping을 기본으로 하고 Z-loss는 off 기준선과 비교한다. 현재는 단일 장치이며 gradient accumulation의 유효 token 수를 합쳐 loss를 정규화한다. 분산 rank 간 정규화는 후속 범위다.

Base → 선택적 DAPT → 분야 SFT(LoRA/DoRA 비교)로 분리한다. p-tuning은 추가 비교 후보이며 base optimizer와 다르다. checkpoint는 CE 외에 한국어 평가와 반복 생성도 확인해 선택한다.
