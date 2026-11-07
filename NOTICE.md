# ライセンスと外部成果物

- このリポジトリの Python コード、設定、ドキュメントは root の `LICENSE`（Apache-2.0）に従う。
- `patches/` は llama.cpp 派生差分であり、`patches/LICENSE`（MIT）と上流の notices を併せて保持する。
- 元モデル `Qwen/Qwen3.8-27B` と公開 QSA BF16/GGUF モデルは Apache-2.0。モデル重みはこの Git リポジトリに含めない。
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) の利用には ODC-By と元データの条件が適用される。学習・評価では固定 revision を指定する。
- DocVQA 評価データは CC BY-NC-SA 4.0（非商用）であり、ここには同梱しない。合成画像生成コードはこのリポジトリの成果物。
- 論文: [On the Design of Qwen3.8-Next Architecture](https://arxiv.org/abs/2608.30320), §2.1.2。

第三者が独自のモデル・データセットへ転用する場合は、それぞれのライセンスを別途確認すること。
