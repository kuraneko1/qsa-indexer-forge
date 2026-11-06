"""Copy a quantized Qwen35 GGUF and append validated FP32 QSA indexers."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest_tensor(tensor):
    return hashlib.sha256(memoryview(tensor.data).cast('B')).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('Run without python -O: validation asserts must remain enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf-python', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--donor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-sha256', required=True)
    parser.add_argument('--donor-sha256', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.gguf_python))
    import gguf

    assert not args.output.exists(), 'Output already exists; refusing to replace it'
    for path, expected in ((args.base, args.base_sha256), (args.donor, args.donor_sha256)):
        assert digest_file(path) == expected, f'Input SHA256 mismatch: {path}'
    base = gguf.GGUFReader(args.base)
    donor = gguf.GGUFReader(args.donor)
    assert base.endianess == donor.endianess
    for key in ('general.architecture', 'qwen35.block_count', 'qwen35.embedding_length',
                'qwen35.attention.head_count', 'qwen35.attention.head_count_kv',
                'qwen35.attention.key_length', 'qwen35.nextn_predict_layers'):
        assert base.get_field(key).contents() == donor.get_field(key).contents(), key
    assert base.get_field('general.architecture').contents() == 'qwen35'
    assert not any('.indexer.' in t.name for t in base.tensors)
    indexers = [t for t in donor.tensors if '.indexer.' in t.name]
    assert len(indexers) == 80
    assert all(t.tensor_type == gguf.GGMLQuantizationType.F32 for t in indexers)
    assert {int(t.name.split('.')[1]) for t in indexers} == set(range(3, 64, 4))
    assert len([t for t in base.tensors if t.name.startswith('blk.64.')]) == 15
    by_name = {t.name: t for t in donor.tensors}
    for t in base.tensors:
        assert t.name in by_name and tuple(t.shape) == tuple(by_name[t.name].shape), t.name

    qsa_keys = ['qwen35.attention.indexer.head_count', 'qwen35.attention.indexer.key_length',
                'qwen35.attention.indexer.top_k', 'qwen35.attention.compress_ratios']
    partial = args.output.with_suffix('.gguf.partial')
    assert not partial.exists(), 'Unfinished output exists; inspect it before retrying'
    writer = gguf.GGUFWriter(partial, arch='qwen35', endianess=base.endianess)
    alignment = base.get_field('general.alignment')
    if alignment:
        writer.data_alignment = alignment.contents()
    for field in base.fields.values():
        if field.name.startswith('GGUF.') or field.name in ['general.architecture'] + qsa_keys:
            continue
        writer.add_key_value(field.name, field.contents(), field.types[0],
                             sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    for key in qsa_keys:
        field = donor.get_field(key)
        assert field is not None, key
        writer.add_key_value(key, field.contents(), field.types[0],
                             sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    writer.add_string('qsa.transplant.base_sha256', args.base_sha256)
    writer.add_string('qsa.transplant.donor_sha256', args.donor_sha256)
    tensors = list(base.tensors) + indexers
    for tensor in tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.n_bytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    original_hashes = {}
    for tensor in tensors:
        original_hashes[tensor.name] = digest_tensor(tensor)
        writer.write_tensor_data(tensor.data, tensor_endianess=base.endianess)
    writer.close()

    output = gguf.GGUFReader(partial)
    assert len(output.tensors) == len(tensors)
    for expected, got in zip(tensors, output.tensors):
        assert expected.name == got.name and expected.tensor_type == got.tensor_type
        assert tuple(expected.shape) == tuple(got.shape)
        assert digest_tensor(got) == original_hashes[got.name], got.name
    for field in base.fields.values():
        if not field.name.startswith('GGUF.'):
            assert field.contents() == output.get_field(field.name).contents(), field.name
    for key in qsa_keys:
        assert donor.get_field(key).contents() == output.get_field(key).contents(), key
    os.rename(partial, args.output)
    report = {
        'status': 'passed', 'base': str(args.base), 'donor': str(args.donor),
        'output': str(args.output), 'base_sha256': args.base_sha256,
        'donor_sha256': args.donor_sha256, 'output_sha256': digest_file(args.output),
        'base_tensors_copied_byte_exact': len(base.tensors),
        'fp32_indexers_copied_byte_exact': len(indexers),
        'mtp_tensors_preserved': sum(t.name.startswith('blk.64.') for t in base.tensors),
        'base_metadata_preserved': True,
        'tensor_types': dict(Counter(t.tensor_type.name for t in output.tensors)),
        'tensor_sha256': original_hashes,
    }
    args.output.with_suffix('.validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'tensor_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
