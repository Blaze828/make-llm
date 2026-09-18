# 아키텍처 조사 기록 · 2026-09-17

후속 보완: 최신 모델 비교보다 저장소에서 읽은 한국어 논문을 우선하는 [한국어 설계](../korean-model-recipe.md)를 추가했다. 아래 최신 구조 조사는 보조 근거이며, tokenizer·optimizer·분야 적응의 현재 계약은 후속 문서를 따른다.

## 조사 범위

기존 저장소 README, 모델 설계, 개발 기획, 논문 정리를 읽고 아래 원 논문·저자 공식 저장소·공개 구현을 확인했다. 관련 문헌 전체를 망라하거나 모든 repository를 실행 검증한 것은 아니다. 원 논문의 주장, 공개 코드에서 확인한 동작, 우리 프로젝트의 선택을 구분한다. 참고 구현을 복사하거나 외부 가중치를 가져오지는 않았다.

GitHub의 `main`과 모델 카드 내용은 바뀔 수 있다. 아래는 조사일 기준 링크이며 commit 고정 스냅샷은 아니다. 구현 단계에는 채택한 소스의 commit과 dependency 버전을 기록해야 한다.

## 논문에서 설계로 연결한 내용

| 자료 | 확인 내용 | 설계 반영 |
|---|---|---|
| [GQA · 2023](https://arxiv.org/abs/2305.13245) | Query head가 KV head를 공유하는 attention | Hq/Hkv=2로 시작; MHA 대조군 유지 |
| [RoFormer · 2021](https://arxiv.org/abs/2104.09864) | 회전 위치 표현 | Q/K에 RoPE, V는 그대로 |
| [RMSNorm · 2019](https://arxiv.org/abs/1910.07467) | RMS 기반 정규화 | Pre-RMSNorm·final RMSNorm |
| [GLU Variants · 2020](https://arxiv.org/abs/2002.05202) | GLU 계열 FFN 비교 | SwiGLU FFN |
| [Qwen3 · 2025, §2](https://arxiv.org/html/2505.09388v1#S2) | Dense 모델의 GQA·RoPE·Pre-RMSNorm·SwiGLU·QK-Norm, 작은 모델의 weight tying | 기본 블록 참고; 폭·층수·어휘는 별도 설계 |
| [DeepSeek-V3 · 2024](https://arxiv.org/abs/2412.19437) | MLA·MoE·multi-token prediction을 결합한 대형 모델 | 대규모 확장 후보; 소형 모델에 바로 이식하지 않음 |
| [DeepSeek-V3.2 · 2025](https://arxiv.org/abs/2512.02556) | DSA를 통한 장문 attention 효율 개선 | 긴 문맥용 대안; 작은 길이에서의 이득 미확인 |
| [Gated Delta Networks · 2024](https://arxiv.org/abs/2412.06464) | gating과 delta rule의 recurrent memory | 하이브리드 mixer 후보 |
| [OlmPool / Cracks in the Foundation](https://allenai.org/papers/olmpool) | 통제된 7B급 비교에서 구조 조합에 따라 장문 확장 차이 관찰 | GQA·QK-Norm을 무조건 이득으로 취급하지 않음 |
| [HyperCLOVA · 2021](https://aclanthology.org/2021.emnlp-main.274/) | 한국어 모델·데이터·토크나이저 연구 | BPE 선택도 한국어 표본으로 검증 |
| [LoRA · 2021](https://arxiv.org/abs/2106.09685) | frozen base 위 low-rank update | 분야 어댑터 분리·고정 projection 이름 |

## GitHub·모델 설정에서 확인한 내용

| 원본 | 확인 범위 | 실제 반영 |
|---|---|---|
| [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) | 공식 README·논문 연결 | Dense/MoE 계열을 구분하고 논문의 블록 설계를 참고 |
| [Transformers Qwen3 구현](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | Q/K/V/O projection, head_dim RMSNorm, norm→RoPE→cache 순서 | 정확한 텐서 계약과 cache 순서 명시 |
| [allenai/OLMo-core](https://github.com/allenai/OLMo-core) | 공개 PyTorch 학습 구성요소와 프로젝트 안내 | 이후 학습·모듈 분리 참고; 이번에는 trainer 미구현 |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | exact attention kernel과 설치·지원 안내 | backend로 분리; 하드웨어 미정이므로 버전 미고정 |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | 선형 attention 및 Gated DeltaNet 구현 제공 | 하이브리드 실험의 후속 구현 후보 |
| [deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | 공식 DSA 실험 모델 안내·논문 연결 | sparse attention의 비교 근거; kernel 실행 안 함 |
| [Qwen3.5 계열 공식 저장소](https://github.com/QwenLM/Qwen3.8) | 조사 시 Qwen3.5 URL이 Qwen3.8로 이동; 최신 계열 안내 | 오래된 Qwen3만을 최신 구조라고 부르지 않음 |
| [Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | text_config의 3개 linear + 1개 full attention 반복, 64층 | 하이브리드 대안 검토; 멀티모달 구성은 본 설계에서 제외 |

Qwen3.8-27B config의 `max_position_embeddings=262144`는 해당 공개 모델의 설정이며, 우리 설계에 그 문맥 처리 능력이 생긴다는 뜻이 아니다. Qwen 계열의 모든 모델이 MoE인 것도 아니다. 하이브리드 attention 여부와 Dense/MoE FFN 여부는 서로 다른 설계 축이다.

## 결론과 불확실성

**지금 이 프로젝트에는 검증 가능한 Dense GQA 기준안을 만들고, 장문·메모리 병목이 확인되면 하이브리드를 비교하는 것이 적절하다는 설계 판단**이다. 공개 모델 간 점수만으로 아키텍처의 우열을 확정할 수 없다. 학습 데이터·토큰 수·후처리·증류 효과가 섞여 있기 때문이다.

기존 50M MHA 설계에서 GQA를 처음부터 명시하고, QK-Norm의 위치·cache 계약·어댑터 호환성·정확한 파라미터 수를 추가했다. GQA=2, 316M 규모, 32K 어휘, 4K 문맥은 논문의 최적값이 아닌 미검증 제안이다. 최종 채택은 [아키텍처 문서](../architecture.md)의 비교 실험 결과로 결정한다.
