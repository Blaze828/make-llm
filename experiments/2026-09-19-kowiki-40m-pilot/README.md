# 2026-09-19 40M 파일럿 실행 기록

실제 Modal 실행에서 받은 작은 메타데이터만 보관한다. 큰 checkpoint와 112개 문서별 평가 출력은 저장소에 복사하지 않았고, 원격 Volume에 남아 있다.

## 실행 정보

- 실행 ID: `kowiki-20260901-40m-pilot`
- Volume: `make-llm-kowiki-20260901`
- 데이터: 2026-09-01 한국어 위키백과 덤프 앞 5,000개 문서
- 분할: train 4,781개, validation 112개, test 104개
- 모델: debug 40M, `korean-pilot-3e-4.json`, sequence length 1,024, micro batch 4, accumulation 8, BF16
- GPU: NVIDIA L4, 총 메모리 22,563 MiB
- 원격 환경: Python 3.13.3, Torch 2.8.0, CUDA 12.8

25단계 sanity run 후 checkpoint를 재개해 총 306 update까지 실행했다.

```powershell
& .\.venv\Scripts\modal.exe run scripts/modal_pilot.py --stage train --steps 306 --stop-after 25 --confirm-cloud
& .\.venv\Scripts\modal.exe run scripts/modal_pilot.py --stage train --steps 306 --confirm-cloud
& .\.venv\Scripts\modal.exe run scripts/modal_pilot.py --stage eval --confirm-cloud
& .\.venv\Scripts\modal.exe run scripts/modal_pilot.py --stage generate --prompt "대한민국의 수도는" --max-new-tokens 64 --confirm-cloud
```

## 학습과 평가

- 학습 상태: `completed`, step 306
- valid tokens seen: 9,144,892
- best validation CE: **4.089396674839877**
- 마지막 train CE: **4.115401263438027**
- 평균 처리량: **11,400.8906 tokens/s** (최소 7,204.4739, 최대 12,645.7004)
- peak allocated GPU memory는 수집하지 않음
- Volume에 `model.pt`, `model.best.pt`, `model.status.json`, `model.events.jsonl` 저장

`model.best.pt`로 validation 112개와 고정 generation prompt 4개를 평가했다. test split은 사용하지 않았다.

- text tokens: 433,430
- CE: **4.089040368683042**
- perplexity: **59.68259093139956**
- bits per byte: **1.8398918042697057**
- generation 평균 repetition: **0.5692234925049725**
- EOS rate: **0.0**
- checkpoint SHA256: `04e1a189b26f0b820f30cad1c1c2d196286b2d63fa06fa9f7a91cac4be490da9`
- suite SHA256: `392c06cfe8879dcc2ed738d4352b7ce829a3c6ec2578f5df001bcc45a17138c9`

고정 generation 4개의 실제 출력은 [`eval-summary.json`](eval-summary.json)에 보관했다. 평가 원본은 원격 Volume의 `evaluation/result/results.json`과 `evaluation/result/predictions.jsonl`에 있다. 대표적으로 `대한민국의 수도는` prompt의 출력은 `대한민국의 배우 배우 김.`으로 이어져 반복이 심했다. 이는 이 작은 파일럿의 실제 출력이며 한국어 능력을 의미하지 않는다.

## 비용

Modal `billing report --show-resources --json`의 해당 App metered cost 합계다. CPU 전처리와 Volume 비용은 제외하며 최종 청구액과 다를 수 있다.

| 작업 | metered cost |
| --- | ---: |
| 25단계 sanity train | 0.04682491 |
| 25→306단계 재개 train | 0.44700640 |
| eval | 0.02283459 |
| 직접 generation | 0.00246521 |
| 합계 | 0.51913111 |

이 결과는 5,000개 문서와 306 update만 사용한 파일럿이다. 최종 모델 품질이나 서비스 가능성을 판단하는 결과가 아니며, BPB는 동일한 문서·토크나이저 조건에서만 비교해야 한다.
