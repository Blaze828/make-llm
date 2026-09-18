# 데이터 확보 전 코드 준비 기록

2026-09-18. 데이터 파이프라인, memmap 로딩, activation checkpointing, 학습 로그·저장 보강, 평가/비교/예산 도구를 추가했다. 상세 범위와 향후 명령은 [준비 문서](../../docs/data-readiness.md)에 있다.

## 이번에 실제 실행한 검사

`python -X utf8 -m scripts.static_check` 결과:

```json
{"static_parsing":"passed","python":45,"json":6,"jsonl_records":10,"training_executed":false}
```

`git diff --check`도 저장소의 기존 줄바꿈 설정에서 통과했다. 별도로 `core.autocrlf=false`를 강제한 검사는 기존 CRLF를 trailing whitespace로 인식해 실패했고, 기존 설정으로 확인했다. 소스 문법 검사와 diff 검사는 실행 정확성을 보장하지 않는다.

## 실행하지 않은 작업

모델 학습, 토크나이저 어휘 학습, 추론, 평가, pytest, GPU 메모리 측정을 실행하지 않았다. 추가한 회귀 테스트 코드도 실행하지 않았다. `MAKE_LLM_ALLOW_TRAINING`은 활성화하지 않았다. 따라서 이번 작업의 loss·정확도·처리량·메모리 측정 결과는 없다.

기존 2026-09-17 실험 수치는 해당 버전의 별도 기록이며 새 코드의 검증 결과로 사용하지 않는다.
