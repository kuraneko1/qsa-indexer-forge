# GGUF utilities

## 量子化済み GGUF に Indexer だけを移植する

`transplant-qsa-gguf.py` は、Indexer を持たない量子化済み `base` と、同じ backbone の QSA `donor` から、80 個の FP32 Indexer tensor と QSA metadata だけを写します。base tensor を再量子化しません。

```sh
python scripts/transplant-qsa-gguf.py \
  --gguf-python /path/to/Qwen3.8-27B-QSA-llama.cpp/gguf-py \
  --base /path/to/base-quantized.gguf \
  --donor /path/to/Qwen3.8-27B-QSA-Q8_0.gguf \
  --output /path/to/new-qsa.gguf \
  --base-sha256 <base-file-sha256> \
  --donor-sha256 <donor-file-sha256>
```

入力 hash、architecture、head / layer 形状、MTP 15 tensor、Indexer 80 tensor を検査します。出力と `.validation.json` が既にある場合の上書きはしません。base と donor は同じモデル構成である必要があります。別の abliterated / fine-tuned backbone に付け替えた場合の品質は、この検査だけでは保証できません。

## 旧 Q8 export の検査

`validate-qsa-gguf.py <root>` は原本の固定 layout を対象にした歴史的検証工具です。

```text
root/models/Qwen3.8-27B-QSA/
root/gguf/Qwen3.8-27B-QSA-Q8_0.gguf
root/gguf/mmproj-Qwen3.8-27B-QSA-F16.gguf
root/logs/
```

HF の FP32 norm を `1 + weight` に fold した GGUF tensor との完全一致、QSA metadata、vision tensor を検査します。任意のファイル名を取る汎用 CLI ではありません。新規移植では `transplant-qsa-gguf.py` が出力する検証 JSON も参照してください。
