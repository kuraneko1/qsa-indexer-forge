# llama.cpp patches

3 つは同じ base の歴史的な独立差分です。**どれか 1 つだけ**を clean な `6209237df53569f07429f3f79fc5354de5fe704f` に適用します。重ね適用はできません。

| patch | 内容 | SHA256 |
|---|---|---|
| `llama-qsa27b.patch` | 初期 mask 参照版 | `b59497f16406ef90d5cebcf73bd2e4b1cc1ad30e913b35113a46dd5d633c128f` |
| `llama-qsa27b-sparse.patch` | 選択 KV を読む疎 CUDA 版。MM2 の画像/MTP cache は未統合 | `4531c8b561fa045041617a2fd0f8a59bab82c7b480ff465087c1b6f4660ab8a8` |
| `llama-qsa27b-mm2-cache.patch` | 推奨。公開 MM2 runtime の全コード差分 | `68f1c9e06077a7ac78b2fedf282aa899fff5f4f88afd7aa4768ee6f98d4e4ca5` |

```sh
git clone https://github.com/pentaCoxian/Qwen-V100-llama.cpp engine
cd engine
git checkout 6209237df53569f07429f3f79fc5354de5fe704f
git apply --check /path/to/qsa-indexer-forge/patches/llama-qsa27b-mm2-cache.patch
git apply /path/to/qsa-indexer-forge/patches/llama-qsa27b-mm2-cache.patch
```

このリポジトリでは base commit の clean archive に対し、3 ファイルとも `git apply --check` が通ることを確認しました。ただし native CUDA の再ビルドと 27B 実モデルの再推論は公開化作業では実行していません。

差分には llama.cpp / ggml 由来コードが含まれるため、[MIT License](LICENSE) と上流 notices を保持してください。
