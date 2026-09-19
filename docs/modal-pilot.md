# Modal에서 40M 파일럿 실행하기

이 문서는 현재 저장소에 데이터·토크나이저·체크포인트가 없다는 전제에서 시작한다. Modal에서 한국어 위키백과의 **2026-09-01 덤프**를 스트리밍해 정제된 첫 5,000개 문서를 만들고, morph32 토크나이저·train/validation 패킹·preflight·짧은 40M 학습·평가·생성을 단계별로 실행한다.

실제 학습 결과와 점수는 실행한 뒤에만 기록할 수 있다. 이 문서는 재실행 가능한 명령을 설명하고, 실제 2026-09-19 실행 결과는 [실행 기록](../experiments/2026-09-19-kowiki-40m-pilot/README.md)에 따로 남긴다.

## 먼저 이해할 용어

- **Modal Function**: 내 컴퓨터가 아니라 Modal의 임시 컨테이너에서 실행되는 함수다.
- **Image**: 그 컨테이너에 설치할 Python과 패키지의 묶음이다. 이 runner는 Python 3.13과 프로젝트에 필요한 패키지를 고정해 사용한다.
- **Volume**: 컨테이너가 종료되어도 파일을 보관하는 Modal의 저장 공간이다. 데이터, 토크나이저, 체크포인트, 로그가 여기에 저장된다.
- **stage**: 전체 작업 중 한 단계다. 한 번에 전부 실행하지 않고 한 단계씩 확인한다.

실행 파일은 [`scripts/modal_pilot.py`](../scripts/modal_pilot.py)다. 기본 Volume 이름은 `make-llm-kowiki-20260901`이고, 기본 실행 ID는 `kowiki-20260901-40m-pilot`이다. 다른 Volume을 쓰려면 `modal run` 전에 `MAKE_LLM_MODAL_VOLUME` 환경 변수를 설정한다.

## 0. 계정·인증·비용을 먼저 확인

Modal GPU를 사용하려면 유효한 결제 수단이 필요하다. Workspace 예산은 월간 사용량 상한이고, spend limit은 실제 지불액 상한이다. 둘을 Dashboard에서 확인한 뒤에만 원격 실행을 승인한다.

공식 안내: [계정 설정](https://modal.com/docs/guide/modal-user-account-setup), [예산](https://modal.com/docs/guide/budgets), [GPU 사용](https://modal.com/docs/guide/gpu), [Volume](https://modal.com/docs/guide/volumes)

이 PC에는 `modal` CLI가 아직 설치되어 있지 않다. 로컬 프로젝트의 `.venv`는 Git에서 무시되므로, 저장소 루트에서 다음처럼 **Modal CLI만** 설치할 수 있다. 이 설치 명령은 원격 작업을 시작하지 않는다.

```powershell
$pilotPython = "C:\Users\PC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $pilotPython -m venv .venv
& .\.venv\Scripts\python.exe -m pip install "modal>=1,<2"
& .\.venv\Scripts\modal.exe setup
& .\.venv\Scripts\modal.exe profile current
& .\.venv\Scripts\modal.exe token info
```

`modal setup`은 브라우저를 열어 로그인하고 토큰을 저장한다. 토큰 ID나 secret을 채팅에 붙여 넣지 않는다. `profile current`와 `token info`가 현재 프로필을 보여주면 인증 확인이 끝난다. 여러 Environment가 있는 계정이면 다음 명령으로 대상 Environment를 명시할 수도 있다.

```powershell
& .\.venv\Scripts\modal.exe config set-environment main
```

## 1. Volume을 만들고 상태를 확인

Volume은 결과물을 저장한다. 아직 계정·예산을 확인하지 않았다면 이 단계도 실행하지 말고 아래 명령만 검토한다.

```powershell
& .\.venv\Scripts\modal.exe volume create make-llm-kowiki-20260901
& .\.venv\Scripts\modal.exe volume list
```

Volume 변경은 명시적으로 `commit()`해야 다른 컨테이너에서 보인다. runner는 각 stage가 끝날 때 commit하고, 시작할 때 reload한다. 파일이 많아지면 Volume의 파일 수와 저장 비용을 확인한다.

## 2. 원격 실행 전에 계획만 보기

SDK 설치와 인자 확인만 하고 싶을 때는 Python으로 `--plan`을 실행한다. 이 경로는 Modal SDK를 import하지 않고 원격 함수·GPU·Modal API를 호출하지 않는다. `modal run ... --plan`도 원격 함수는 실행하지 않지만, CLI가 App을 읽고 인증할 수 있으므로 완전한 오프라인 확인에는 아래 Python 명령을 사용한다.

```powershell
& .\.venv\Scripts\python.exe scripts/modal_pilot.py --stage download --plan
```

모든 stage는 `--confirm-cloud`가 없으면 원격 호출을 거부한다. 이 확인값은 계정 인증과 비용 상한을 사람이 검토했다는 뜻이지, 비용을 보장하는 기능은 아니다.

## 3. 단계별 실행

아래 순서대로 한 단계씩 실행한다. 각 단계가 끝나면 Volume의 `<run-id>/stages/<stage>.json` 완료 표식과 로그를 확인한다. 실패한 중간 산출물이 있으면 같은 ID로 덮어쓰지 않고 새 `--run-id`를 사용한다.

```powershell
$modal = ".\.venv\Scripts\modal.exe"
$confirm = "--confirm-cloud"

& $modal run scripts/modal_pilot.py --stage download  $confirm
& $modal run scripts/modal_pilot.py --stage prepare   $confirm
& $modal run scripts/modal_pilot.py --stage tokenizer $confirm
& $modal run scripts/modal_pilot.py --stage pack      $confirm
& $modal run scripts/modal_pilot.py --stage preflight $confirm
& $modal run scripts/modal_pilot.py --stage suite     $confirm
```

각 단계의 뜻은 다음과 같다.

1. `download`: 날짜가 고정된 위키백과 URL에서 필요한 앞부분만 스트리밍해 5,000개 문서를 만든다. 전체 덤프를 로컬이나 Volume에 저장하지 않는다.
2. `prepare`: 라이선스·문서 길이·깨진 텍스트·중복을 검사하고 hash 기반으로 train/validation/test를 나눈다. test는 이후 suite에 사용하지 않는다.
3. `tokenizer`: train split만 사용해 MeCab 경계 기반 byte BPE 32K를 학습한다. 32K를 만들지 못하면 다음 단계로 넘어가지 않는다.
4. `pack`: 같은 토크나이저로 train과 validation을 길이 1024의 binary row로 만든다.
5. `preflight`: 모델 어휘 크기, sequence length, 배열 크기, checksum을 읽기 전용으로 확인한다. 실패하면 학습하지 않는다.
6. `suite`: validation 문서 전체와 고정 generation prompt로 새 평가 파일을 만든다. 예전 로컬 평가 파일을 사용하지 않는다.

## 4. 짧은 학습과 재개

기본값은 40M 모델, 306 optimizer update, micro batch 4, accumulation 8, sequence length 1024, BF16이다. 기존 파일럿의 비교 가능한 학습 설정인 `korean-pilot-3e-4.json`을 사용한다. GPU 함수는 L4 한 장, 최대 45분, 자동 retry 0회로 제한된다. 실행이 중간에 끊겨도 25 update마다 체크포인트가 저장되도록 했다.

처음에는 비용과 메모리를 확인할 수 있게 25 update만 실행한다.

```powershell
& $modal run scripts/modal_pilot.py --stage train --steps 306 --stop-after 25 $confirm
```

로그와 `model.pt.status.json`이 `stopped`이고 `model.pt`가 있으면, 같은 명령에서 `--stop-after`만 빼고 다시 실행한다. runner가 기존 checkpoint를 찾아 `--resume`으로 이어간다.

```powershell
& $modal run scripts/modal_pilot.py --stage train --steps 306 $confirm
```

정상 종료 후에만 `stages/train.json`이 생긴다. 306 update까지 완료되지 않았는데 평가를 실행할 수 없도록 막아 두었다. 최대 update는 코드에서 1,000으로 제한되어 있다.

## 5. 평가와 생성

평가는 가장 좋은 validation CE를 기록한 checkpoint가 있으면 그것을, 없으면 마지막 checkpoint를 사용한다. 결과는 CE·perplexity·bits-per-byte와 generation 결과를 포함한다.

```powershell
& $modal run scripts/modal_pilot.py --stage eval $confirm
& $modal run scripts/modal_pilot.py --stage generate --prompt "대한민국의 수도는" --max-new-tokens 64 $confirm
```

평가 결과는 `evaluation/result/results.json` 요약과 `evaluation/result/predictions.jsonl` 문서별 결과로 저장하고, 직접 생성 결과는 `generation/*.json`에 저장된다. 생성 문자열은 모델이 실제로 출력한 값만 기록한다.

Volume에 저장된 파일은 필요할 때 다음처럼 받을 수 있다. 체크포인트처럼 큰 파일은 CLI `volume get`을 사용한다.

```powershell
& $modal volume ls make-llm-kowiki-20260901
& $modal volume get make-llm-kowiki-20260901 kowiki-20260901-40m-pilot/evaluation/result/results.json .\results.json
```

## 제한과 기록 규칙

- 실제 다운로드·학습·평가 수치와 비용은 사용자가 위 명령을 실행한 뒤에만 기록한다. 이 저장소 문서에 예상 loss나 속도를 결과처럼 적지 않는다.
- 위키백과 이용 조건은 현재 프로젝트 설정의 `cc-by-sa-4.0`을 사용한다. 실제 이용·공개 조건은 [데이터 출처 기록표](data-sources.md)에 확인일과 근거를 채운 뒤 결정한다.
- test split은 보관만 하고 평가 suite에는 넣지 않는다. validation 문서와 고정 prompt만 평가에 사용한다.
- Modal Volume은 원격 저장소다. 삭제 명령은 이 runner에 포함하지 않았으며, 원본 덤프 전체를 저장하지 않는다.
- 이 runner는 단일 프로세스 학습만 지원한다. GPU·CUDA·Python·주요 패키지 버전은 학습 시작 시 run root의 `environment.json`과 checkpoint metadata에 기록한다.
