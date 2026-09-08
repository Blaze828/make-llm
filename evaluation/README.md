# evaluation

성능 측정. 비교 실험의 근거가 여기서 나온다. (README 11, 12번)

| 파일 | 역할 |
|---|---|
| `perplexity.py` | 한국어 / 도메인 perplexity |
| `korean_bench.py` | 한국어 능력 유지 확인 (Catastrophic Forgetting 측정) |
| `traffic_bench.py` | 교통 도메인 성능 |
| `cost.py` | 학습 시간, GPU 메모리, 저장 용량, 학습 파라미터 수 |

목표는 `교통 성능 ↑↑` 이면서 `한국어 성능 →` 인 지점을 찾는 것.
