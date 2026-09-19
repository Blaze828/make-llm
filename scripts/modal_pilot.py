"""Staged, bounded Modal runner for the 40M Korean Wikipedia pilot.

This file intentionally does not start a Modal Function at import time.  Use
``python scripts/modal_pilot.py --stage download --plan`` for an offline plan,
then pass ``--confirm-cloud`` only after authentication and billing limits are
ready.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.request

APP_NAME = "make-llm-kowiki-40m-pilot"
VOLUME_NAME = os.environ.get("MAKE_LLM_MODAL_VOLUME", "make-llm-kowiki-20260901")
VOLUME_MOUNT = "/mnt/artifacts"
REMOTE_PROJECT = "/root/project"
DUMP_DATE = "20260901"
DEFAULT_RUN_ID = "kowiki-20260901-40m-pilot"
MAX_DOCUMENTS = 5000
DEFAULT_STEPS = 306
MAX_STEPS = 1000
GPU_TIMEOUT_SECONDS = 45 * 60
GPU_TYPE = "L4"
STAGE_NAMES = ("download", "prepare", "tokenizer", "pack", "preflight", "suite", "train", "eval", "generate", "status")


def _safe_run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("run-id must contain only letters, numbers, '.', '_' or '-'")
    return value


def _offline_plan():
    """Print a plan without importing the Modal SDK or contacting Modal."""
    parser = argparse.ArgumentParser(description="Print a local-only Modal pilot plan")
    parser.add_argument("--stage", choices=STAGE_NAMES, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args, _ = parser.parse_known_args()
    _safe_run_id(args.run_id)
    print(f"volume: {os.environ.get('MAKE_LLM_MODAL_VOLUME', VOLUME_NAME)}")
    print(f"run-id: {args.run_id}")
    print(f"stage: {args.stage}")
    print("원격 함수·GPU·Modal SDK를 실행하지 않았습니다 (--plan)")
    print("계정·인증·예산을 확인한 뒤 modal run ... --confirm-cloud를 사용하세요.")


if __name__ == "__main__" and "--plan" in sys.argv:
    _offline_plan()
    raise SystemExit(0)


import modal

ROOT = Path(__file__).resolve().parents[1]


def _image():
    # Pin the remote Python and core packages.  The source repository is
    # mounted at startup so changing Python code does not require a rebuild.
    return (
        modal.Image.debian_slim(python_version="3.13")
        .uv_pip_install(
            "torch==2.8.0",
            "tokenizers==0.23.2",
            "mecab-ko==1.0.2",
            "mecab-ko-dic==1.0.0",
            "numpy==2.3.2",
        )
        .workdir(REMOTE_PROJECT)
        .add_local_dir(
            str(ROOT),
            remote_path=REMOTE_PROJECT,
            copy=False,
            ignore=[
                ".git",
                ".venv",
                "datasets",
                "checkpoints",
                "experiments",
                "__pycache__",
            ],
        )
    )


image = _image()
app = modal.App(name=APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _paths(run_id):
    run_id = _safe_run_id(run_id)
    root = Path(VOLUME_MOUNT) / run_id
    return {
        "root": root,
        "source": root / "source",
        "raw": root / "source" / f"kowiki-{DUMP_DATE}-first{MAX_DOCUMENTS}.jsonl",
        "clean": root / "data" / "clean",
        "tokenizer": root / "tokenizer" / "morph32",
        "packed": root / "packed",
        "preflight": root / "preflight.json",
        "suite": root / "evaluation" / "suite.jsonl",
        "evaluation": root / "evaluation" / "result",
        "generation": root / "generation",
        "train": root / "train",
        "stages": root / "stages",
    }


def _marker(paths, stage):
    return paths["stages"] / f"{stage}.json"


def _done(paths, stage):
    return _marker(paths, stage).is_file()


def _write_marker(paths, stage, **details):
    from data_pipeline.common import write_json

    write_json(
        _marker(paths, stage),
        {"stage": stage, "completed_at_utc": _now(), **details},
    )


def _require_done(paths, stage):
    if not _done(paths, stage):
        raise RuntimeError(f"먼저 '{stage}' 단계를 완료하세요: {_marker(paths, stage)}")


def _run(command, *, env=None):
    print("$ " + shlex.join([str(item) for item in command]), flush=True)
    subprocess.run(command, cwd=REMOTE_PROJECT, env=env, check=True)


def _training_env():
    env = os.environ.copy()
    env["MAKE_LLM_ALLOW_TRAINING"] = "1"
    return env


def _write_environment_manifest(paths):
    """Record the actual remote runtime next to the checkpoint."""
    import importlib.metadata
    import platform
    import torch

    gpu = None
    gpu_memory_bytes = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = properties.name
        gpu_memory_bytes = properties.total_memory
    package_versions = {}
    for package in ("torch", "tokenizers", "mecab-ko", "mecab-ko-dic", "numpy", "modal"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    from data_pipeline.common import write_json

    write_json(
        paths["root"] / "environment.json",
        {
            "recorded_at_utc": _now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "requested_gpu": GPU_TYPE,
            "gpu_name": gpu,
            "gpu_memory_bytes": gpu_memory_bytes,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "package_versions": package_versions,
        },
    )


def _source_stage(paths, dump_date, max_documents):
    if _done(paths, "download"):
        return {"status": "already-complete", "path": str(paths["raw"])}
    if paths["raw"].exists():
        raise RuntimeError("download 산출물이 일부 존재하지만 완료 표식이 없습니다. 새 run-id를 사용하세요.")
    if dump_date != DUMP_DATE:
        raise ValueError(f"이 파일럿은 고정 덤프 날짜 {DUMP_DATE}만 허용합니다.")
    if max_documents != MAX_DOCUMENTS:
        raise ValueError(f"이 재현 파일럿은 정확히 {MAX_DOCUMENTS}개 문서만 사용합니다.")

    from data_pipeline.wiki_dump import convert
    from data_pipeline.common import sha256_file, write_json

    source = f"kowiki-{dump_date}"
    url = f"https://dumps.wikimedia.org/kowiki/{dump_date}/kowiki-{dump_date}-pages-articles.xml.bz2"
    paths["source"].mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "make-llm-modal-pilot/1.0 (research pipeline)"},
    )
    print(f"Downloading and streaming only the first {max_documents} kept articles from {url}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        stats = convert(
            response,
            paths["raw"],
            source,
            "cc-by-sa-4.0",
            "korean_general",
            max_documents,
            200,
        )
    if stats.get("kept") != max_documents:
        raise RuntimeError(f"덤프에서 목표 문서 수를 확보하지 못했습니다: {stats}")
    write_json(
        paths["source"] / "source.json",
        {
            "source": source,
            "dump_date": dump_date,
            "url": url,
            "license": "cc-by-sa-4.0",
            "max_documents": max_documents,
            "retrieved_at_utc": _now(),
            "conversion": "data_pipeline.wiki_dump.convert; min_chars=200",
            "stats": stats,
            "jsonl_sha256": sha256_file(paths["raw"]),
        },
    )
    _write_marker(paths, "download", source=source, url=url, stats=stats)
    return {"status": "completed", "stats": stats}


def _prepare_stage(paths):
    if _done(paths, "prepare"):
        return {"status": "already-complete", "manifest": str(paths["clean"] / "manifest.json")}
    _require_done(paths, "download")
    if paths["clean"].exists() and any(paths["clean"].iterdir()):
        raise RuntimeError("prepare 산출물이 일부 존재하지만 완료 표식이 없습니다. 새 run-id를 사용하세요.")
    from data_pipeline.prepare import prepare

    config_path = Path(REMOTE_PROJECT) / "configs" / "data" / "preparation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = prepare([paths["raw"]], paths["clean"], config)
    _write_marker(paths, "prepare", stats=manifest["stats"], manifest=str(paths["clean"] / "manifest.json"))
    return {"status": "completed", "stats": manifest["stats"]}


def _tokenizer_stage(paths):
    if _done(paths, "tokenizer"):
        return {"status": "already-complete", "path": str(paths["tokenizer"])}
    _require_done(paths, "prepare")
    if paths["tokenizer"].exists() and any(paths["tokenizer"].iterdir()):
        raise RuntimeError("tokenizer 산출물이 일부 존재하지만 완료 표식이 없습니다. 새 run-id를 사용하세요.")
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "tokenizer.morph_bpe",
        "--corpus-manifest",
        str(paths["clean"] / "manifest.json"),
        "--output",
        str(paths["tokenizer"]),
        "--config",
        f"{REMOTE_PROJECT}/configs/tokenizer/korean-morph-bpe-32k.json",
        "--vocab-size",
        "32000",
    ]
    _run(command, env=_training_env())
    metadata = json.loads((paths["tokenizer"] / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("actual_vocab_size") != 32000:
        raise RuntimeError(
            "5,000개 파일럿에서 32K 어휘를 만들지 못했습니다. "
            f"actual_vocab_size={metadata.get('actual_vocab_size')}"
        )
    _write_marker(paths, "tokenizer", metadata=metadata)
    return {"status": "completed", "vocab_size": metadata["actual_vocab_size"]}


def _pack_stage(paths):
    if _done(paths, "pack"):
        return {"status": "already-complete"}
    _require_done(paths, "prepare")
    _require_done(paths, "tokenizer")
    for split in ("train", "validation"):
        output = paths["packed"] / split
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"{split} pack 산출물이 일부 존재하지만 완료 표식이 없습니다. 새 run-id를 사용하세요.")
        _run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-m",
                "data_pipeline.pack",
                "--corpus-manifest",
                str(paths["clean"] / "manifest.json"),
                "--split",
                split,
                "--tokenizer",
                str(paths["tokenizer"]),
                "--output",
                str(output),
                "--sequence-length",
                "1024",
            ]
        )
    _write_marker(paths, "pack", splits=["train", "validation"], sequence_length=1024)
    return {"status": "completed", "sequence_length": 1024}


def _preflight_stage(paths):
    if _done(paths, "preflight"):
        return {"status": "already-complete", "path": str(paths["preflight"])}
    _require_done(paths, "pack")
    from scripts.preflight import inspect

    result = inspect(
        f"{REMOTE_PROJECT}/configs/architecture/debug-40m.json",
        f"{REMOTE_PROJECT}/configs/data/preparation.json",
        paths["packed"] / "train" / "manifest.json",
        paths["packed"] / "validation" / "manifest.json",
    )
    from data_pipeline.common import write_json

    write_json(paths["preflight"], result)
    if result["artifact_checks"] != "passed":
        raise RuntimeError("preflight가 차단되었습니다. preflight.json의 blockers를 확인하세요.")
    _write_marker(paths, "preflight", result=result)
    return {"status": "completed", "result": result}


def _suite_stage(paths):
    if _done(paths, "suite"):
        return {"status": "already-complete", "path": str(paths["suite"])}
    _require_done(paths, "prepare")
    if paths["suite"].exists():
        raise RuntimeError("평가 suite가 일부 존재하지만 완료 표식이 없습니다. 새 run-id를 사용하세요.")
    from data_pipeline.common import read_jsonl, sha256_file

    validation_path = paths["clean"] / "validation.jsonl"
    paths["suite"].parent.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(validation_path))
    if not rows:
        raise RuntimeError("validation split이 비어 있어 평가 suite를 만들 수 없습니다.")
    # Use every validation document.  This keeps the evaluation result tied to
    # the complete grouped validation split rather than an arbitrary sample.
    selected = rows
    records = [
        {
            "id": f"validation-text-{row['id']}",
            "type": "text",
            "domain": row.get("domain", "korean_general"),
            "text": row["text"],
        }
        for row in selected
    ]
    prompts = [
        "대한민국의 수도는",
        "인공지능은 사람의",
        "한국의 계절은 보통",
        "다음 문장을 이어 쓰세요. 우리나라는",
    ]
    records.extend(
        {
            "id": f"fixed-generation-{index}",
            "type": "generation",
            "domain": "korean_general",
            "prompt": prompt,
        }
        for index, prompt in enumerate(prompts, 1)
    )
    with paths["suite"].open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_marker(
        paths,
        "suite",
        path=str(paths["suite"]),
        records=len(records),
        validation_source_sha256=sha256_file(validation_path),
        test_split_used=False,
    )
    return {"status": "completed", "records": len(records), "test_split_used": False}


def _train_stage(paths, steps, stop_after):
    _require_done(paths, "preflight")
    if _done(paths, "train"):
        return {"status": "already-complete", "checkpoint": str(paths["train"] / "model.pt")}
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between 1 and {MAX_STEPS}")
    if stop_after is not None and not 1 <= stop_after <= steps:
        raise ValueError("stop-after must be between 1 and steps")
    _write_environment_manifest(paths)
    paths["train"].mkdir(parents=True, exist_ok=True)
    checkpoint = paths["train"] / "model.pt"
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "training.pretrain",
        "--train-manifest",
        str(paths["packed"] / "train" / "manifest.json"),
        "--validation-manifest",
        str(paths["packed"] / "validation" / "manifest.json"),
        "--tokenizer",
        str(paths["tokenizer"]),
        "--recipe",
        f"{REMOTE_PROJECT}/configs/training/korean-pilot-3e-4.json",
        "--model-config",
        f"{REMOTE_PROJECT}/configs/architecture/debug-40m.json",
        "--steps",
        str(steps),
        "--batch-size",
        "4",
        "--accumulation",
        "8",
        "--sequence-length",
        "1024",
        "--device",
        "cuda",
        "--bf16",
        "--gradient-checkpointing",
        "--save-every",
        "25",
        "--eval-every",
        "25",
        "--output",
        str(checkpoint),
    ]
    if checkpoint.exists():
        command += ["--resume", str(checkpoint)]
    if stop_after is not None:
        command += ["--stop-after", str(stop_after)]
    _run(command, env=_training_env())
    status_path = checkpoint.with_suffix(".status.json")
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if status.get("status") == "completed":
        _write_marker(paths, "train", status=status, checkpoint=str(checkpoint))
    return {"status": status.get("status", "unknown"), "checkpoint": str(checkpoint), "details": status}


def _eval_stage(paths):
    _require_done(paths, "train")
    _require_done(paths, "suite")
    if _done(paths, "eval"):
        return {"status": "already-complete", "path": str(paths["evaluation"])}
    checkpoint = paths["train"] / "model.best.pt"
    if not checkpoint.exists():
        checkpoint = paths["train"] / "model.pt"
    if not checkpoint.exists():
        raise RuntimeError("학습 checkpoint가 없습니다.")
    if paths["evaluation"].exists() and any(paths["evaluation"].iterdir()):
        raise RuntimeError("평가 결과가 일부 존재하지만 완료 표식이 없습니다. 새 run-id를 사용하세요.")
    _run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "evaluation.run",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(paths["tokenizer"]),
            "--suite",
            str(paths["suite"]),
            "--output",
            str(paths["evaluation"]),
            "--max-new-tokens",
            "64",
            "--device",
            "cuda",
        ]
    )
    result_path = paths["evaluation"] / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _write_marker(paths, "eval", checkpoint=str(checkpoint), results=result)
    return {"status": "completed", "results": result}


def _generation_stage(paths, prompt, max_new_tokens):
    _require_done(paths, "train")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not 1 <= max_new_tokens <= 128:
        raise ValueError("max-new-tokens must be between 1 and 128")
    checkpoint = paths["train"] / "model.best.pt"
    if not checkpoint.exists():
        checkpoint = paths["train"] / "model.pt"
    if not checkpoint.exists():
        raise RuntimeError("학습 checkpoint가 없습니다.")
    paths["generation"].mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    output_path = paths["generation"] / f"{digest}.json"
    if output_path.exists():
        return {"status": "already-complete", "path": str(output_path)}
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "training.generate",
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(paths["tokenizer"]),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        "cuda",
    ]
    print("$ " + shlex.join([str(item) for item in command]), flush=True)
    result = subprocess.run(
        command,
        cwd=REMOTE_PROJECT,
        env=_training_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    from data_pipeline.common import write_json

    write_json(
        output_path,
        {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "checkpoint": str(checkpoint),
            "generated_text": result.stdout,
        },
    )
    return {"status": "completed", "path": str(output_path)}


def _status_stage(paths):
    completed = sorted(path.stem for path in paths["stages"].glob("*.json")) if paths["stages"].is_dir() else []
    return {"run_id": paths["root"].name, "volume": VOLUME_NAME, "completed_stages": completed}


def _execute_cpu(stage, run_id, dump_date, max_documents):
    paths = _paths(run_id)
    if stage == "download":
        return _source_stage(paths, dump_date, max_documents)
    if stage == "prepare":
        return _prepare_stage(paths)
    if stage == "tokenizer":
        return _tokenizer_stage(paths)
    if stage == "pack":
        return _pack_stage(paths)
    if stage == "preflight":
        return _preflight_stage(paths)
    if stage == "suite":
        return _suite_stage(paths)
    if stage == "status":
        return _status_stage(paths)
    raise ValueError(f"CPU stage not supported: {stage}")


def _execute_gpu(stage, run_id, steps, stop_after, prompt, max_new_tokens):
    paths = _paths(run_id)
    if stage == "train":
        return _train_stage(paths, steps, stop_after)
    if stage == "eval":
        return _eval_stage(paths)
    if stage == "generate":
        return _generation_stage(paths, prompt, max_new_tokens)
    raise ValueError(f"GPU stage not supported: {stage}")


@app.function(
    volumes={VOLUME_MOUNT: volume},
    timeout=30 * 60,
    retries=0,
)
def run_cpu_stage(stage, run_id, dump_date, max_documents):
    volume.reload()
    try:
        result = _execute_cpu(stage, run_id, dump_date, max_documents)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        volume.commit()


@app.function(
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TYPE,
    timeout=GPU_TIMEOUT_SECONDS,
    retries=0,
)
def run_gpu_stage(stage, run_id, steps, stop_after, prompt, max_new_tokens):
    volume.reload()
    try:
        result = _execute_gpu(stage, run_id, steps, stop_after, prompt, max_new_tokens)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        volume.commit()


STAGES = STAGE_NAMES


def _parser():
    parser = argparse.ArgumentParser(description="Bounded, staged Modal runner for the 40M pilot")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dump-date", default=DUMP_DATE)
    parser.add_argument("--max-documents", type=int, default=MAX_DOCUMENTS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--prompt", default="대한민국의 수도는")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--plan", action="store_true", help="Print the staged commands; never contact Modal")
    parser.add_argument("--confirm-cloud", action="store_true", help="Required before any remote invocation")
    return parser


def _print_plan(args):
    print(f"volume: {VOLUME_NAME}")
    print(f"run-id: {args.run_id}")
    print(f"stage: {args.stage}")
    print("원격 실행 없음 (--plan)")
    print("다음 단계: Modal 로그인·예산 확인 후 같은 명령에 --confirm-cloud를 추가하세요.")


@app.local_entrypoint()
def main(*raw_args):
    args = _parser().parse_args(list(raw_args))
    _safe_run_id(args.run_id)
    if args.plan:
        _print_plan(args)
        return
    if not args.confirm_cloud:
        raise SystemExit(
            "원격 실행을 막았습니다. 먼저 modal setup과 비용 한도를 확인한 뒤 "
            "--confirm-cloud를 명시하세요. 준비만 하려면 --plan을 사용하세요."
        )
    if args.stage in {"download", "prepare", "tokenizer", "pack", "preflight", "suite", "status"}:
        result = run_cpu_stage.remote(args.stage, args.run_id, args.dump_date, args.max_documents)
    else:
        if args.steps > MAX_STEPS:
            raise SystemExit(f"steps is capped at {MAX_STEPS}")
        result = run_gpu_stage.remote(
            args.stage,
            args.run_id,
            args.steps,
            args.stop_after,
            args.prompt,
            args.max_new_tokens,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Modal normally calls the local entrypoint.  This guard gives a clear
    # message instead of silently starting a cloud job when run with Python.
    print("Use: modal run scripts/modal_pilot.py --stage status --plan")
