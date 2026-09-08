# datasets

데이터 수집·정제. (README 4번)

```
raw/         원본 (git 제외)
processed/   정제 결과 (git 제외)
scripts/     수집·정제 코드 (git 포함)
```

## 반드시 확인할 것

중복 제거 / 깨진 문장 제거 / 광고·스팸 제거 / 개인정보 제거 /
이상한 문자 제거 / 라이선스 확인 / 평가 문제와 겹치는 데이터 제거

## 버전을 반드시 남긴다

```
korean_dataset_v1
korean_dataset_v2
```

어떤 데이터로 어떤 모델을 만들었는지 나중에 추적할 수 있어야 한다.
데이터 버전은 `experiments/`의 각 실험 기록에 적는다.
