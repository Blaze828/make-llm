"""Read configuration and artifact metadata only; never imports torch or trains."""
import argparse
import json
from pathlib import Path
from data_pipeline.common import artifact_path, sha256_file


def inspect(model_path, preparation_path, train_path=None, validation_path=None):
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    prep = json.loads(Path(preparation_path).read_text(encoding="utf-8"))
    blockers = []
    if not prep.get("allowed_licenses"):
        blockers.append("실제 데이터 이용 조건 확인 후 allowed_licenses를 지정해야 합니다.")
    if not train_path or not validation_path:
        blockers.append("train/validation 토큰 데이터 manifest가 아직 지정되지 않았습니다.")
    manifests = []
    sizes = {"input_ids":4,"document_ids":8,"attention_mask":1}
    for split, path in (("train",train_path),("validation",validation_path)):
        if not path: continue
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(data)
        if data.get("kind") != "packed_tokens" or data.get("format_version") != 1:
            blockers.append(f"{split}: 지원하지 않는 manifest 형식")
            continue
        if data["split"] != split: blockers.append(f"{split}: split 불일치")
        if data["vocab_size"] != model["vocab_size"]: blockers.append(f"{split}: 어휘 크기 불일치")
        if not 2 <= data["sequence_length"] <= model["max_position_embeddings"]:
            blockers.append(f"{split}: 문맥 길이 불일치")
        if data["rows"] <= 0 or data["valid_targets"] <= 0: blockers.append(f"{split}: 비어 있는 데이터")
        for key, size in sizes.items():
            info = data["arrays"][key]
            array = artifact_path(path.parent,info["path"])
            if not array.is_file():
                blockers.append(f"{split}/{key}: 파일 없음")
            elif array.stat().st_size != data["rows"]*data["sequence_length"]*size or sha256_file(array) != info["sha256"]:
                blockers.append(f"{split}/{key}: 크기 또는 checksum 불일치")
    if len(manifests)==2:
        for key in ("tokenizer_sha256","corpus_manifest_sha256","sequence_length"):
            if manifests[0].get(key) != manifests[1].get(key): blockers.append(f"train/validation {key} 불일치")
    return {"artifact_checks":"blocked" if blockers else "passed", "blockers":blockers,
            "training_executed":False,
            "not_verified":["실제 데이터 품질과 평가 오염", "GPU 실행과 메모리", "새 코드의 동작 및 수치 검증", "총 토큰 예산과 분야별 비율"],
            "note":"이 검사는 학습 시작 승인이나 성능 검증을 의미하지 않습니다."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config",default="configs/architecture/base-316m.json")
    parser.add_argument("--preparation-config",default="configs/data/preparation.json")
    parser.add_argument("--train-manifest")
    parser.add_argument("--validation-manifest")
    args = parser.parse_args()
    result = inspect(args.model_config,args.preparation_config,args.train_manifest,args.validation_manifest)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["artifact_checks"]=="passed" else 2)


if __name__ == "__main__": main()
