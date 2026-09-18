# adapters

현재 `lora.py`에 LoRA와 DoRA Linear wrapper·attach·활성화 전환·artifact 저장/복원·호환성 검사를 구현했다. SFT CLI는 `training/finetune.py`이며 `--method lora|dora`로 선택한다. 아래 `qlora.py` 등 개별 파일 구성은 초기 계획이고 4-bit backend는 미구현이다. [실행 안내](../docs/implementation.md).

Base 모델을 고정한 채 붙이는 재학습 모듈. (README 8, 9번)

| 파일 | 역할 |
|---|---|
| `lora.py` | 저랭크 행렬만 학습. Rank 8/16/32/64 실험 대상 |
| `qlora.py` | Base를 저비트로 양자화 + LoRA |
| `dora.py` | LoRA 개선 방식 |

Base 모델 코드와 반드시 분리한다. 어댑터를 갈아끼워도 `model/`은 손대지 않는다.

```
Korean Base LLM
      │
  ┌───┼───┐
Traffic  Finance  Code
 LoRA     LoRA    LoRA
```
