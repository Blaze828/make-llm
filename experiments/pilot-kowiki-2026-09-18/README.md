# 파일럿 사전학습 실행 기록 — 한국어 위키백과

실행일: 2026-09-18 · 기록일: 2026-09-19 · 환경: B (`inozosa`, RTX 4070 SUPER)

[기획서 §7 단계 B: 파일럿 사전학습](../../docs/project-plan.md)을 실행했다. **파이프라인 전 구간을 실제 데이터로 처음 완주했다.** 목적은 한국어 품질이 아니라 정제·처리량·loss 확인과 토크나이저 결정이며, 기획서가 정한 완료 기준은 `보류 loss 개선·비용 추정`이다.

이 기록은 실제 실행한 명령과 출력에 근거한다. 실행하지 않은 항목은 §8에 적었다.

## 1. 데이터 출처

| 항목 | 값 |
|---|---|
| 출처 | 한국어 위키백과 덤프 |
| URL | `https://dumps.wikimedia.org/kowiki/20260901/kowiki-20260901-pages-articles.xml.bz2` |
| 압축 크기 | 1,365,272,177 bytes (전체) |
| 실제 사용 | 스트림 앞부분에서 기사 5,000건까지만 소비 |
| `source` 값 | `kowiki-20260901` |
| `license` 값 | `cc-by-sa-4.0` |

`latest`가 아닌 **날짜가 박힌 URL**을 사용했다. `latest`는 매달 바뀌어 재현이 되지 않는다.

`configs/data/preparation.json`의 `allowed_licenses`를 `["cc-by-sa-4.0"]`로 설정했다. 위키백과 본문의 널리 알려진 라이선스에 근거했으며, **법적 검토를 받은 것은 아니다.** [데이터 출처 문서 §6](../../docs/data-sources.md)의 확인 기록표에 사람이 확인한 날짜와 확인자를 채워야 한다. 동일조건변경허락(SA)이 학습 가중치 공개 조건에 미치는 영향은 아직 정하지 않았다.

## 2. 덤프 변환 (`data_pipeline/wiki_dump.py`, 신규)

```powershell
curl -s "https://dumps.wikimedia.org/kowiki/20260901/kowiki-20260901-pages-articles.xml.bz2" | `
  python -X utf8 -m data_pipeline.wiki_dump --input - --output datasets/raw/kowiki-20260901.jsonl `
  --source kowiki-20260901 --license cc-by-sa-4.0 --max-documents 5000
```

| 결과 | 값 |
|---|---:|
| 소요 시간 | **15.7초** |
| 검사한 기사 | 5,790 |
| 너무 짧아 제외 | 790 |
| 정제 후 빈 문서 | 0 |
| 출력 문서 | **5,000** |
| 출력 문자 | 25,014,174 |
| 출력 UTF-8 bytes | 57,349,748 |
| 입력 파일 sha256 | `165d6ac6e8c84b5d…` |

위키텍스트는 MediaWiki 파서가 아니라 패턴 규칙으로 제거한다. 틀 전개·표 내용·정보상자 값은 렌더링하지 않고 버리므로 **본문 산문만 남는다.** 이름공간 0 이외와 넘겨주기는 제외했다.

### 변환 중 발견해 고친 결함 2건

두 결함 모두 5,000문서 규모에서만 드러났고, 50문서 표본에서는 보이지 않았다.

1. **닫히지 않은 `[[` / `{{` 처리** — 원본 위키텍스트에는 `[[DNA 삼중나선]_`처럼 잘못 닫힌 마크업이 있다. 초기 구현은 이때 문서의 나머지 전체를 미처리 상태로 내보내거나(링크) 통째로 삭제했다(틀). 그 결과 **19개 문서에 `[[` 잔여물 3,167건**이 몰렸다. 열림 토큰만 버리고 나머지는 계속 처리하도록 고쳐 **잔여물 0건**이 되었다.
2. **`<[^>]+>` 규칙이 수식 부등호를 삭제** — `0 < x > y` 같은 텍스트에서 ` < x >` 구간이 지워졌다. 태그 모양(`</?[a-zA-Z][a-zA-Z0-9]*…>`)만 매칭하도록 좁혀 부등호를 보존한다.

회귀 테스트는 `tests/test_wiki_dump.py`에 추가했다 (5개).

### 남은 잔여 마크업 (5,000문서 기준)

| 종류 | 건수 / 문서 수 |
|---|---:|
| 내부링크 `[[` | 0 / 0 |
| ref 태그 | 0 / 0 |
| 외부링크 | 0 / 0 |
| 틀 `{{` | 141 / 5 |
| 표 `{|` | 44 / 7 |
| `분류:` | 36 / 10 |
| `파일:` | 3 / 2 |

전체 2,500만 문자 중 약 20개 문서의 흔적 수준이다. 완전 제거는 하지 않았고, MediaWiki 파서를 쓰지 않는 한 남는다.

## 3. 정제 (`data_pipeline.prepare`)

| 결과 | 값 |
|---|---:|
| 소요 시간 | **5분 11초** |
| 입력 | 5,000 |
| 거절: 반복 문자 | 2 |
| 거절: 깨진 문자 | 1 |
| 완전 중복 제거 | **0** |
| 유사 중복 연결 | **0** |
| train 문서 / bytes | 4,781 / 54,386,835 |
| validation 문서 / bytes | 112 / 1,389,705 |
| test 문서 / bytes | 104 / 1,484,756 |

분할 비율은 validation 2.24%, test 2.08%로 설정값 2%/2%의 hash 기반 기대치에 부합했다. 위키백과 기사끼리는 중복이 없어 dedup 경로는 통과만 확인했고 **실제 중복 제거 능력은 검증되지 않았다.**

**처리 속도가 이번 파일럿의 가장 중요한 비용 발견이다.** 57MB를 5분 11초에 처리했다 (약 184KB/s, 16문서/s). MinHash 8밴드의 문서별 shingle 해싱이 Python 루프에서 지배적이다. 단순 선형 외삽하면 **1GB 정제에 약 1.5시간**이 필요하다. 이 외삽은 측정이 아니며, 문서 길이 분포와 후보 검증량에 따라 달라진다.

## 4. 토크나이저 학습과 비교

네 종류를 같은 train split(4,781문서)으로 학습했다. 어휘 학습은 train만 사용했고 각 실행은 약 30초였다. **모두 목표 vocab에 도달**했으므로 이 코퍼스는 48K merge에도 충분하다. MeCab fallback 문서는 **0건**이었다.

| 이름 | family | 요청 → 실제 vocab | sha256 |
|---|---|---|---|
| morph16 | morph_byte_bpe | 16,000 → 16,000 | `01b27fa7050774da…` |
| morph32 | morph_byte_bpe | 32,000 → 32,000 | `653c056828118943…` |
| morph48 | morph_byte_bpe | 48,000 → 48,000 | `24bbddf5eaad5b2e…` |
| plain32 | byte_bpe (대조군) | 32,000 → 32,000 | `134aa6c657dc66ed…` |

MeCab-ko 1.0.2 / mecab-ko-dic 1.0.0, 사전 sha256 `a55bb95e5cfd…`.

### 압축 효율 (validation 112문서, `tokenizer.benchmark`)

| 토크나이저 | vocab | tokens/byte | 문자/토큰 | 복원 실패 | 총 토큰 |
|---|---:|---:|---:|---:|---:|
| morph16 | 16,000 | 0.32292 | 1.363 | 0 | 448,769 |
| morph32 | 32,000 | 0.31189 | 1.411 | 0 | 433,430 |
| morph48 | 48,000 | 0.30682 | 1.435 | 0 | 426,392 |
| **plain32** | 32,000 | **0.20051** | **2.195** | 0 | **278,653** |

복원 실패는 전 토크나이저 0건이다. 형태소 계열은 vocab을 3배 늘려도 압축이 5%만 개선된다(0.3229 → 0.3068).

**형태소 계열이 압축에서 크게 졌다.** plain32는 같은 텍스트를 **36% 적은 토큰**으로 표현한다. 원인은 설계상 의도된 것이다 — `tokenizer/bpe.py`가 형태소 조각을 별도 iterator 항목으로 넘겨 **형태소 경계를 넘는 merge를 막기** 때문에 토큰이 짧아진다. 같은 패킹에서도 확인된다: 동일 train split이 morph32에서 17,803 row, plain32에서 11,509 row가 되었다.

## 5. 패킹과 실행 전 점검

문맥 1,024, `data_pipeline.pack`. 두 경로 모두 26초 내에 끝났다.

| 경로 | train rows | train 유효 target | validation rows | validation 유효 target |
|---|---:|---:|---:|---:|
| morph32 (`tokens-v1`) | 17,803 | 16,632,682 | 460 | 433,542 |
| plain32 (`tokens-plain`) | 11,509 | 10,342,906 | 301 | 278,765 |

`scripts.preflight`가 `debug-40m` 설정과 morph32 경로에 대해 **`passed`, 차단 사유 0건, 종료 코드 0**을 반환했다. 이 검사가 통과한 것은 이번이 처음이다.

## 6. 학습 3 run

공통: `configs/architecture/debug-40m.json` (**39,986,688** 파라미터), 306 optimizer update, micro batch 4 × 문맥 1,024 × accumulation 8 = update당 32,768 할당 토큰, BF16 autocast, CUDA, seed 42, `--save-every 100 --eval-every 50`, activation checkpointing 미사용.

| run | 토크나이저 | peak LR | 유효 토큰 | train CE 첫→끝 | **최저 val CE** | val PPL |
|---|---|---:|---:|---|---:|---:|
| `pilot-lr3e-4` | morph32 | 3e-4 | 9,144,892 | 10.6885 → 4.1159 | **4.0899** | 59.73 |
| `pilot-lr1e-4` | morph32 | 1e-4 | 9,144,892 | 10.6885 → 4.5705 | 4.5292 | 92.68 |
| `pilot-plain32` | plain32 | 3e-4 | 8,801,651 | 10.4776 → 6.6436 | 6.5952 | 731.60 |

세 run 모두 `status: completed`, 비유한 loss·OOM·중단 없음. 마지막 gradient norm은 0.52~0.76으로 clipping 임계값 1.0 아래였다.

### validation CE 추이 (50 update 간격)

| step | 50 | 100 | 150 | 200 | 250 | 300 | 306 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lr3e-4` | 5.0431 | 4.6043 | 4.3613 | 4.2188 | 4.1383 | 4.0934 | 4.0899 |
| `lr1e-4` | 5.8034 | 5.0198 | 4.7590 | 4.6358 | 4.5682 | 4.5320 | 4.5292 |
| `plain32` | 7.9712 | 7.4014 | 6.9805 | 6.7688 | 6.6557 | 6.6002 | 6.5952 |

**세 run 모두 단조 개선했고 악화 구간이 없었다.** 기획서의 파일럿 완료 기준인 `보류 loss 개선`을 충족한다. 306 update는 학습 곡선이 평탄해지기 전이므로 **수렴값이 아니다.**

### 학습률 비교 (기획서 §7 단계 B 요구사항)

`3e-4`와 `1e-4`를 같은 데이터·같은 토큰 예산·같은 seed로 비교했다. 유효 토큰 수가 9,144,892로 동일하므로 통제된 비교다. **3e-4가 명확히 우수하다** (val CE 4.0899 vs 4.5292). 두 값 모두 발산하지 않았으므로 더 높은 학습률을 시도할 여지가 남아 있으나, 이번 파일럿에서는 확인하지 않았다.

### 실측 처리량과 메모리

| 항목 | 값 |
|---|---|
| `tokens_per_second` 중앙값 | **16,794 / 17,163 / 17,103** (run 순서대로) |
| 범위 | 11,915 ~ 19,730 |
| update당 소요 | 약 1.9초 |
| GPU 메모리 (프로세스, `nvidia-smi`) | **8,040 MiB / 12,282 MiB** |

이 값은 **환경 B 전용**이다. 환경 A(RTX 3060)에서는 다르게 측정된다 ([환경 문서](../../docs/environments.md)). 메모리는 프로세스 수준 관측값이며 `torch.cuda.max_memory_allocated` 값이 아니다. `debug-40m`은 12GB에서 문맥 1,024·micro batch 4를 activation checkpointing 없이 소화했다.

이제 `scripts.plan_budget --measured-tokens-per-second`에 넣을 실측값이 생겼다. 이전에는 `hours_at_user_measured_target_tokens_per_second`가 항상 `null`이었다.

## 7. 토크나이저 판정 — BPB 비교

토크나이저가 다르면 token CE/PPL을 직접 비교할 수 없다([데이터 준비 문서 §5](../../docs/data-readiness.md)). 두 모델의 최저 validation CE를 같은 분모(validation 112문서의 1,389,705 UTF-8 bytes)로 환산했다.

```
BPB = CE_per_token × 평가 토큰 수 / (UTF-8 bytes × ln 2)
```

| 모델 | val 토큰 | CE | PPL | **BPB** |
|---|---:|---:|---:|---:|
| morph32 `lr3e-4` | 433,542 | 4.0899 | 59.73 | **1.8407** |
| plain32 `lr3e-4` | 278,765 | 6.5952 | 731.60 | 1.9086 |
| morph32 `lr1e-4` | 433,542 | 4.5292 | 92.68 | 2.0385 |

**압축과 품질이 반대 방향이다.**

- 압축률: plain32가 36% 우수
- 동일 compute의 BPB: **morph32가 3.6% 우수** (1.8407 vs 1.9086)

plain32는 같은 306 update 안에서 **morph32의 약 1.50배 텍스트**(약 43.9MB vs 29.3MB, validation 기준 tokens/byte로 환산)를 보았는데도 BPB가 더 나빴다. 즉 형태소 경계 제약이 압축을 희생하는 대가로 바이트당 모델링에서는 이득을 냈다.

**이 BPB는 packed validation의 CE에서 산술 환산한 값이며 `evaluation.run`으로 측정한 값이 아니다.** 아래 한계를 감안해야 한다.

- 40M 모델, 306 update, 900만 토큰 규모의 결과다. 316M·1B에서 같은 순서가 유지된다는 근거는 없다.
- seed 1개, plain32는 학습률 1개(3e-4)만 시도했다.
- 두 run의 유효 target 수가 9,144,892 vs 8,801,651로 3.8% 다르다.
- 3.6%는 작은 차이이며 seed 간 분산을 측정하지 않았다.

따라서 **`configs/tokenizer/korean-morph-bpe-32k.json`의 형태소 인지 32K 기본값을 바꿀 근거는 없고, 확정할 근거도 아직 없다.** 문맥 길이당 담기는 텍스트량을 중시한다면 plain 계열을 다시 검토할 수 있다.

## 8. 실행하지 않은 것

- **`evaluation.run` / `evaluation.compare`** — 평가 묶음(`datasets/processed/eval/validation-suite.jsonl`, 본문 112 + 생성 5, sha256 `49780bbb7525881a…`)은 만들었으나 실행하지 않았다. 따라서 생성 반복률·EOS 비율·객관식·QA 점수는 **없다.**
- **`base-316m` GPU 실측** — 환경 B에서 316M 모델의 메모리·처리량을 측정하지 않았다. 12GB에서 문맥 1,024 학습이 가능한지 미확인.
- 생성 샘플 확인. 이 모델이 한국어 문장을 만들 수 있는지 보지 않았다.
- SFT, LoRA/DoRA, 교통 특화, 중단·재개 실험.
- morph16 / morph48의 모델 학습 (압축 비교만 했다).
- test split은 **봉인 상태**를 유지했다. 토크나이저 결정에 validation만 사용했다.
- 환경 A에서의 교차 확인.

## 9. 산출물 위치

가중치·토크나이저·데이터는 `.gitignore` 대상이므로 저장소에 없다. 환경 B 로컬에만 있다.

| 산출물 | 경로 |
|---|---|
| 원문 JSONL | `datasets/raw/kowiki-20260901.jsonl` |
| 정제 코퍼스 | `datasets/processed/clean-v1/` (manifest sha256 `4d4af05a90743993…`) |
| 토큰 데이터 | `datasets/processed/tokens-v1/`, `datasets/processed/tokens-plain/` |
| 평가 묶음 | `datasets/processed/eval/validation-suite.jsonl` |
| 체크포인트 | `checkpoints/pilot-{lr3e-4,lr1e-4,plain32}.pt` (+ `.best.pt`, `.events.jsonl`, `.status.json`) |
| 토크나이저 | `checkpoints/tok-{morph-16k,morph-32k,morph-48k,plain-32k}/` |
| 토크나이저 비교 | `tokenizer-comparison.json` (이 디렉터리, 저장소에 포함) |

## 10. 다음 단계

1. `evaluation.run`으로 세 모델을 평가해 생성 거동(반복률·EOS)을 확인하고, 산술 환산한 BPB를 실제 측정값으로 대체한다.
2. `base-316m`을 환경 B에서 1 update만 돌려 메모리·처리량을 실측하고, 12GB에서 문맥 1,024가 가능한지 판정한다.
3. `scripts.plan_budget --measured-tokens-per-second 17000`으로 본 사전학습 기간을 산정한다.
4. 정제 속도가 본 학습 규모의 병목인지 판단한다. 필요하면 MinHash 경로를 개선한다.
5. 모두의 말뭉치·AI Hub 승인을 받아 위키백과 외 출처를 추가한다 ([데이터 출처 문서](../../docs/data-sources.md)).
6. 환경 A에서 같은 파이프라인을 실행해 기능 결과가 일치하는지 확인한다.
