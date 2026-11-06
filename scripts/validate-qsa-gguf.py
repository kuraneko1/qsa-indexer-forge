"""Validate the native export metadata and every FP32 indexer tensor."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from safetensors import safe_open
from gguf import GGUFReader
root = Path(sys.argv[1])
model = root / 'models/Qwen3.8-27B-QSA'
r = GGUFReader(root / 'gguf/Qwen3.8-27B-QSA-Q8_0.gguf')
tensors = {t.name:t for t in r.tensors}
qsa = json.loads((model / 'config.json').read_text())['qsa']
assert r.get_field('general.architecture').contents() == 'qwen35'
assert r.get_field('qwen35.attention.indexer.head_count').contents() == qsa['indexer_heads']
assert r.get_field('qwen35.attention.indexer.key_length').contents() == qsa['indexer_head_dim']
assert r.get_field('qwen35.attention.indexer.top_k').contents() == qsa['budget']
count = 0
with safe_open(model/'qsa-indexers.safetensors',framework='numpy') as source:
    for name in source.keys():
        prefix, suffix = name.split('._qsa_indexer.')
        il = int(prefix.split('.')[3])
        dst = {'q_proj.weight':'q_proj.weight', 'k_proj.weight':'k_proj.weight',
               'q_norm':'q_norm.weight', 'k_norm':'k_norm.pre_weight',
               'k_block_norm':'k_norm.weight'}[suffix]
        want = source.get_tensor(name)
        if suffix.endswith('norm'): want = want + np.float32(1)
        got = tensors[f'blk.{il}.indexer.{dst}']
        assert int(got.tensor_type) == 0
        np.testing.assert_array_equal(got.data.reshape(want.shape),want)
        count += 1
assert count == sum('.indexer.' in name for name in tensors) == 80
v = GGUFReader(root/'gguf/mmproj-Qwen3.8-27B-QSA-F16.gguf')
assert len(v.tensors) == 334
report = {'indexer_tensors_verified_exactly':count,
          'indexer_norm_scale_fold':'1 + original FP32 weight',
          'main_tensor_types':dict(Counter(str(t.tensor_type) for t in r.tensors)),
          'vision_tensor_count':len(v.tensors),
          'main_bytes':(root/'gguf/Qwen3.8-27B-QSA-Q8_0.gguf').stat().st_size,
          'vision_bytes':(root/'gguf/mmproj-Qwen3.8-27B-QSA-F16.gguf').stat().st_size}
(root/'logs/gguf-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
