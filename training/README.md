# training

사전학습과 재학습 루프.

| 파일 | 역할 |
|---|---|
| `dataloader.py` | 토큰 배열 → 배치 (x, y). y는 x를 한 칸 민 것 |
| `pretrain.py` | Base LLM 사전학습 |
| `finetune.py` | 도메인 재학습 (Full FT / LoRA / QLoRA / DoRA) |

하이퍼파라미터는 코드에 박지 않고 `configs/`에서 읽는다. (README 18번)
