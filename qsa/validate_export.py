"""Validate an exported 27B model's complete tensor layout without loading it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import torch
from transformers import AutoConfig, AutoModelForImageTextToText


def validate_export(path):
    path = Path(path)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    headers, total_bytes = {}, 0
    for filename in sorted(set(index["weight_map"].values())):
        file = path / filename
        with file.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        last_offset = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in headers or index["weight_map"].get(name) != filename:
                raise ValueError(f"Duplicate or incorrectly mapped tensor: {name}")
            headers[name] = info
            total_bytes += info["data_offsets"][1] - info["data_offsets"][0]
            last_offset = max(last_offset, info["data_offsets"][1])
        if file.stat().st_size != 8 + size + last_offset:
            raise ValueError(f"Incomplete safetensors shard: {filename}")
    if set(headers) != set(index["weight_map"]):
        raise ValueError("The weight map does not match the actual tensors")
    if total_bytes != index["metadata"]["total_size"]:
        raise ValueError("Incorrect tensor payload size in the weight index")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    if config.model_type != "qwen3_5_qsa" or config.qsa["multimodal_policy"] != "dense":
        raise ValueError("Incorrect model type or multimodal policy")
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(
            config, trust_remote_code=True, attn_implementation="sdpa"
        )
    expected = model.state_dict()
    missing = sorted(set(expected) - set(headers))
    unexpected = sorted(set(headers) - set(expected))
    if missing or any(not key.startswith("mtp.") for key in unexpected):
        raise ValueError(f"Missing tensors: {missing}; unexpected tensors: {unexpected}")
    for name, tensor in expected.items():
        if list(tensor.shape) != headers[name]["shape"]:
            raise ValueError(f"Wrong shape: {name}")
        if "._qsa_indexer." in name and headers[name]["dtype"] != "F32":
            raise ValueError(f"Indexer must remain FP32: {name}")
    return {
        "tensor_layout_validated": True,
        "tensor_count": len(headers),
        "runtime_tensor_count": len(expected),
        "preserved_mtp_tensors": len(unexpected),
        "tensor_payload_bytes": total_bytes,
        "full_27b_forward": False,
        "multimodal_quality_benchmark": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = validate_export(args.model)
    if args.write_report:
        (Path(args.model) / "structural_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
