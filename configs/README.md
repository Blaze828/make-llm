# configs

실행 상태: 구조 JSON은 `ModelConfig.load()`가 검증·로드한다. tokenizer CLI는 어휘 크기·special 목록을, 학습 CLI는 optimizer·scheduler·Z-loss·clipping을 사용한다. 실제 장치·precision·배치·총 steps는 CLI로 지정한다. recipe의 null 자원 값은 자동 설정되지 않는다. 기존 설계 설명보다 [실행 안내](../docs/implementation.md)를 우선한다.

## 한국어 tokenizer·학습 recipe

- [korean-morph-bpe-32k.json](tokenizer/korean-morph-bpe-32k.json): MeCab 기반 어휘 학습, 실제 인코딩은 ByteLevel, 총 32K ID.
- [korean-base.json](training/korean-base.json): AdamW·cosine·weight decay 그룹·Z-loss 비교·평가 계약. 파일 참조는 저장소 루트 기준이다.

실행기는 지원 필드를 읽고, 실제 분석기/사전 버전과 입력 데이터 hash를 artifact에 기록한다. GPU 배치와 총 steps는 CLI로 지정한다. 전체 명세의 모든 선택적 최적화를 구현한 것은 아니다. 출처·프로젝트 제안·미확정 항목은 [한국어 LLM 설계](../docs/korean-model-recipe.md)에서 구분한다.

## 아키텍처 v1 설정

아래 JSON은 자체 `ModelConfig`와 `KoreanLM`으로 실행한다. Transformers에 바로 전달하는 config 형식은 아니다. 공통 필드 의미·계산식은 [아키텍처 문서](../docs/architecture.md)에 정의했다.

| 파일 | 역할 | 파라미터 수 |
|---|---|---:|
| [debug-40m.json](architecture/debug-40m.json) | 정합성·작은 학습 흐름 확인 | 39,986,688 |
| [base-316m.json](architecture/base-316m.json) | 한국어 기본 모델 설계 기준 | 315,936,768 |
| [scale-1.2b.json](architecture/scale-1.2b.json) | 학습 자원 확보 후 확장 | 1,198,104,576 |

모든 설정은 별도 JSON 한 개로 완결되며 override 순서가 없다. `qk_norm_affine=shared_across_heads`는 Q용/K용 scale 각각 head_dim개를 공유한다는 뜻이다. `rotary_fraction=1.0`은 Q/K의 head_dim 전체에 회전을 적용한다. `max_position_embeddings`는 목표 상한이며 학습 완료를 의미하지 않는다.

토크나이저 ID, dtype/backend, optimizer, batch size, LoRA rank는 구조 설정과 별도로 관리한다. tokenizer 생성 후 전체 어휘 크기가 vocab_size와 같은지 확인해야 한다. 구조를 변경하면 새 config hash를 발급하고 checkpoint·adapter 호환성을 검사한다.

## 향후 학습 설정 예시

하이퍼파라미터를 코드에서 분리한다. (README 18번)

값을 코드에 직접 적으면 실험할 때마다 코드를 고쳐야 하고,
어떤 값으로 어떤 결과가 나왔는지 추적할 수 없다.

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

실험마다 config 파일을 따로 두고, `experiments/`의 기록에서 그 파일명을 참조한다.
