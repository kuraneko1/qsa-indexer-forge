# Figure assets

3 図とも [architecture-explainer](https://github.com/kuraneko1/architecture-explainer) の **mid**、`jade-violet` palette、衝突検査付き renderer で作成しました。

| Figure | PNG | Editable SVG | Scene JSON |
|---|---|---|---|
| Training to llama.cpp | `qsa-training-flow-mid.png` | `qsa-training-flow-mid.svg` | `qsa-training-flow-mid.scene.json` |
| llama.cpp pipeline | `llama-cpp-pipeline.png` | `llama-cpp-pipeline.svg` | `llama-cpp-pipeline.scene.json` |
| Dense / QSA evaluation | `evaluation-flow.png` | `evaluation-flow.svg` | `evaluation-flow.scene.json` |

## Regenerate

`architecture-explainer` を取得し、Python に Matplotlib・Pillow・Noto Sans CJK JP を用意します。

```sh
python /path/to/architecture-explainer/scripts/diagram.py \
  docs/assets/llama-cpp-pipeline.scene.json \
  --out /tmp/llama-cpp-pipeline --palette jade-violet
```

renderer は PNG、editable / outlined SVG、preview、geometry、validation JSON を出力します。テキストや矢印の衝突、未解決の配線交差がある場合は出力を拒否します。

図の事実の出典は `qsa-indexer-forge` revision `eb0fd3feee754ff119d7ccc0c613558c60876436`。詳細と評価条件は各図を参照している Markdown 文書に記載しています。
