# make-llm
Building an LLM

## 1. 최종 목표

한국어를 잘하는 기본 LLM을 만든 후, 새로운 분야의 데이터를 추가로 학습해서 빠르게 특화 모델을 만들 수 있는 구조


# 2. 가장 먼저 알아야 할 것

직접 정해야 하는 값

- 레이어 수
- Hidden Size
- Attention Head 수
- Vocabulary 크기
- Learning Rate
- Batch Size
- 학습 데이터 양
- 학습 데이터 비율
- LoRA Rank
- Context Length

=> 하이퍼파라미터

처음 값을 정할 때는 논문이나 기존 모델을 참고하고, 이후 실험하면서 수정


# 3. 전체 개발 순서

## 먼저 논문과 기존 모델 구조를 조사

이미 검증된 모델을 참고

### 꼭 볼 것

1. Attention Is All You Need
   - Transformer 기본 구조 이해

2. OLMo / OLMo 2
   - LLM을 실제로 어떻게 학습하는지 보기 좋음

3. HyperCLOVA
   - 한국어 특화 LLM 참고

4. Polyglot-Ko
   - 한국어 데이터와 한국어 모델 참고

5. LoRA
   - 빠른 재학습을 위한 핵심 방법

6. QLoRA
   - GPU 메모리를 더 적게 사용하면서 재학습

7. DoRA
   - LoRA 개선 방식

8. Don't Stop Pretraining
   - 특정 분야에 특화시키는 방법 참고


### 읽은 논문 정리

읽은 논문의 핵심 질문, 구조와 수치, 학습 방법, 관련 용어는 [논문 정리](docs/papers/README.md)에 정리한다.


논문을 볼 때는 아래만 정리해도 충분

```text
왜 이 방법을 썼는가?
모델 크기는?
레이어 수는?
Hidden Size는?
Learning Rate는?
Tokenizer는?
데이터는 얼마나 사용했나?
학습은 어떻게 했나?
어떤 실험을 했나?
무엇이 가장 성능에 영향을 줬나?
```


# 4. 한국어 데이터 준비

한국어 LLM의 성능은 모델 구조만큼 데이터가 중요

한국어 데이터
├─ 일반 문서
├─ 뉴스
├─ 위키
├─ 책
├─ 블로그
├─ 대화문
├─ 논문
├─ 과학/수학
├─ 코드
└─ 한국 문화/역사


## 반드시 확인할 것

- 중복 데이터 제거
- 깨진 문장 제거
- 광고/스팸 제거
- 개인정보 제거
- 이상한 문자 제거
- 라이선스 확인
- 평가 문제와 겹치는 데이터 제거

데이터는 반드시 버전

```text
korean_dataset_v1
korean_dataset_v2
korean_dataset_v3
```

그래야 어떤 데이터로 어떤 모델을 만들었는지 나중에 알 수 있다.

---

# 5. 한국어 Tokenizer 만들기

Tokenizer는 문장을 모델이 이해할 수 있는 숫자로 나누는 역할

예:

```text
대한민국의 인공지능 산업
```

Tokenizer에 따라

```text
대한 / 민국 / 의 / 인 / 공 / 지 / 능
```

처럼 너무 잘게 나뉠 수도 있고,

```text
대한민국 / 의 / 인공지능 / 산업
```

처럼 적당히 나뉠 수 있음

한국어를 너무 잘게 쪼개면 학습이 비효율적

## 실험할 것

```text
BPE 16K
BPE 32K
BPE 48K
```

정도로 비교

확인할 것은:

- 한국어 한 문장을 몇 token으로 나누는지
- 영어 처리
- 숫자 처리
- 한글 + 영어 혼합 문장
- 전문용어
- 신조어

Base LLM을 완성한 뒤에는 Tokenizer를 함부로 바꾸지 않는 것이 좋음

---

# 6. 아주 작은 LLM부터 만들기

처음부터 1B 모델을 만들면 안됨

먼저 작은 모델로 코드가 제대로 작동하는지 확인

추천 순서:

```text
10M ~ 30M
↓
50M ~ 100M
↓
300M
↓
가능하면 500M ~ 1B
```

## 첫 번째 작은 모델에서 확인할 것

- Loss가 정상적으로 내려가는가?
- 문장을 조금이라도 생성하는가?
- Attention이 정상인가?
- Causal Mask가 정상인가?
- Gradient가 정상인가?
- NaN이 발생하지 않는가?
- 저장 후 다시 불러올 수 있는가?

첫 모델의 목표는 성능이 아니라

코드가 제대로 만들어졌는지 확인하는 것

# 7. Base LLM 구조

처음에는 복잡하게 만들지 않기

추천 구조:

```text
Tokenizer
↓
Embedding
↓
Transformer Block
   ├─ RMSNorm
   ├─ Self Attention
   ├─ RoPE
   ├─ SwiGLU
   └─ Residual Connection
↓
여러 Layer 반복
↓
LM Head
↓
다음 Token 예측
```

처음에는 검증된 구조를 그대로 따라가는 것이 좋음

새로운 구조를 실험하는 것은 Base 모델이 정상적으로 학습된 뒤에 하기


# 8. 재학습이 쉬운 구조로 만든다

이 프로젝트의 핵심이다.

Base LLM 전체를 매번 다시 학습하지 않기

추천 구조:

```text
               Korean Base LLM
                     │
       ┌─────────────┼─────────────┐
       │             │             │
   Traffic LoRA   Finance LoRA   Code LoRA
       │             │             │
       ↓             ↓             ↓
   교통 특화       금융 특화      코드 특화
```

Base Model은 그대로 두고 작은 추가 모듈만 학습

이렇게 하면:

- GPU 메모리 감소
- 저장 용량 감소
- 재학습 비용 감소
- Base 모델 보호
- 분야별 모델 관리 쉬움

이라는 장점.

# 9. 재학습 방법

## 1. Full Fine-Tuning

모델 전체를 다시 학습

장점:
- 성능이 가장 높을 가능성이 있음

단점:
- 느림
- GPU 많이 필요
- 기존 한국어 능력이 망가질 수 있음

## 2. LoRA

모델 전체는 고정하고 작은 추가 가중치만 학습

장점:
- 메모리 적게 사용
- 저장 공간 적음
- 분야별로 Adapter를 따로 저장 가능

졸업작품에서 가장 중요하게 볼 방법


## 3. QLoRA

Base 모델을 더 작은 bit로 저장하고 LoRA를 학습

장점:
- GPU 메모리를 더 적게 사용할 수 있음

GPU가 부족하면 특히 유용하다.


## 4. DoRA

LoRA를 개선한 방식

LoRA와 비교 실험하기 좋음

### 한국어 특화 LLM을 만든 후 졸업 작품에 만든 LLM을 사용해서 재학습시킬 경우
Ex) 교차로 통행량 분석(또는 교차로 통행량을 실시간으로 분석)하여 신호등 제어
# 10. 교통 분야를 첫 번째 재학습 대상으로 사용

Base LLM을 만든 다음 교통 데이터를 학습

## 교통 데이터 예시

- 교통공학 자료
- 교차로 신호 운영
- 통행량 자료
- 도로 관련 문서
- 교통 관련 논문
- SUMO 자료
- 교통량 분석 데이터

구조:

```text
Korean Base LLM
↓
Traffic Dataset
↓
LoRA / QLoRA / DoRA
↓
Traffic LLM
```

예를 들어:

```text
사용자:
북쪽 대기 차량 30대
동쪽 대기 차량 8대
현재 북쪽 방향 적색 80초

현재 교차로 상황을 분석해줘.

모델:
북쪽 방향의 대기 차량이 많고 적색 시간이 길어
해당 방향의 신호 우선순위를 높일 필요가 있습니다.
```

이런 식으로 교통 상황을 이해하도록 만들 수 있음

# 11. 가장 중요한 실험

단순히 "잘 됩니다"라고 하면 안 됨

반드시 비교

## 실험 A

```text
Base LLM
```

## 실험 B

```text
Base LLM + Full Fine-Tuning
```

## 실험 C

```text
Base LLM + LoRA
```

## 실험 D

```text
Base LLM + QLoRA
```

## 실험 E

```text
Base LLM + DoRA
```

그리고 아래 항목을 비교

| 항목 | Full FT | LoRA | QLoRA | DoRA |
|---|---|---|---|---|
| 교통 성능 | | | | |
| 한국어 성능 유지 | | | | |
| 학습 시간 | | | | |
| GPU 메모리 | | | | |
| 추가 저장 용량 | | | | |
| 학습 가능한 Parameter 수 | | | | |


# 12. 반드시 확인해야 하는 문제: 기존 능력 손실

교통 데이터를 너무 많이 학습하면 원래 잘하던 한국어를 못하게 될 수 있음

예:

```text
Base Model

한국어 성능: 80
교통 성능: 30
```

교통 학습 후:

```text
Traffic Model

한국어 성능: 55
교통 성능: 85
```

이러면 교통 성능은 좋아졌지만 한국어 능력이 너무 떨어짐

이 문제를 Catastrophic Forgetting

그래서 목표는:

```text
교통 성능 ↑↑
한국어 성능 →
```

이어야 한다.


# 13. 기존 능력을 유지하는 방법도 실험

교통 데이터만 학습하지 않고 일반 한국어 데이터를 조금 섞어보기

예:

```text
실험 1
교통 100%
```

```text
실험 2
교통 90%
일반 한국어 10%
```

```text
실험 3
교통 80%
일반 한국어 20%
```

그리고 어떤 비율이 가장 좋은지 비교


# 14. LoRA에서도 실험해야 하는 값

LoRA도 아무 값이나 사용하면 안됨

예:

```text
Rank = 8
Rank = 16
Rank = 32
Rank = 64
```

비교한다.

Rank가 높으면 표현력이 늘어날 수 있지만:

- 학습 Parameter 증가
- 메모리 증가
- 저장 용량 증가

가 발생

따라서 성능과 비용을 같이


# 15. 하이퍼파라미터 정하는 방법

처음부터 감으로 정하지 않는다.

순서:

```text
논문 조사
↓
비슷한 크기의 모델 값 확인
↓
초기값 선정
↓
작은 모델에서 실험
↓
좋은 범위 찾기
↓
큰 모델에 적용
```

예:

Learning Rate 실험

```text
1e-4
3e-4
5e-4
7e-4
```

결과가 5e-4 근처에서 좋다면:

```text
4e-4
5e-4
6e-4
```

로 다시 좁혀보기

이렇게

# 16. 절대 큰 모델에서 먼저 실험하지 않기

예를 들어 1B 모델로 Learning Rate 10개를 비교하면 비용이 너무 큼

그래서:

```text
30M
↓
많은 실험

100M
↓
다시 확인

300M
↓
최종 확인

500M ~ 1B
↓
본 학습
```

순서가 좋다.

---

# 17. 코드 구조도 중요하다

추천 구조:

```text
project/
│
├─ tokenizer/
│
├─ model/
│   ├─ attention.py
│   ├─ mlp.py
│   ├─ rope.py
│   ├─ norm.py
│   └─ transformer.py
│
├─ adapters/
│   ├─ lora.py
│   ├─ qlora.py
│   └─ dora.py
│
├─ training/
│
├─ evaluation/
│
├─ datasets/
│
├─ configs/
│
├─ checkpoints/
│
└─ experiments/
```

Base Model과 Adapter 코드를 분리하는 것이 중요


# 18. Config 파일 사용

코드 안에 숫자를 직접 적지 않는 것이 좋음

예:

```yaml
model:
  hidden_size: 512
  layers: 8
  heads: 8

training:
  learning_rate: 0.0005
  batch_size: 32

lora:
  rank: 16
  alpha: 32
```

이렇게 하면 실험할 때 코드 수정 없이 값만 바꿀 수 있음
---

# 19. 실험 기록

모든 실험에 번호

예:

```text
EXP-001
Base 30M
Learning Rate 3e-4

EXP-002
Base 30M
Learning Rate 5e-4

EXP-003
LoRA Rank 8

EXP-004
LoRA Rank 16
```

각 실험마다 저장할 것:

- 모델 설정
- 데이터 버전
- Learning Rate
- Batch Size
- 학습 시간
- GPU 사용량
- Train Loss
- Validation Loss
- 한국어 성능
- Domain 성능
