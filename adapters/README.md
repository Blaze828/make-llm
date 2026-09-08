# adapters

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
