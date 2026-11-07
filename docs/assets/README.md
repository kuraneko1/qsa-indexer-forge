# Figure assets

工程図は `jade-violet` palette のベクター描画で、アイコンも画像ではなく図形として描いています。`scripts/render_flowchart.py` が文字・矢印の衝突を検査し、問題があれば書き出しません。

| Figure | PNG | Editable SVG | Scene JSON |
|---|---|---|---|
| 全体像 | `qsa-overview.png` | `qsa-overview.svg` | `qsa-overview.scene.json` |
| Training to llama.cpp | `qsa-training-flow-mid.png` | `qsa-training-flow-mid.svg` | `qsa-training-flow-mid.scene.json` |
| llama.cpp pipeline | `llama-cpp-pipeline.png` | `llama-cpp-pipeline.svg` | `llama-cpp-pipeline.scene.json` |
| Dense / QSA evaluation | `evaluation-flow.png` | `evaluation-flow.svg` | `evaluation-flow.scene.json` |
| 変換前 / 変換後のアーキテクチャ | `architecture/qsa-architecture-ja.png` | `architecture/qsa-architecture-ja.svg`（outlined 版・PDF も同梱） | 原研究リポジトリの `build.py` で再生成 |

English 版は `architecture/qsa-architecture-en.png` / `.svg` / `-outlined.svg` / `.pdf` です。

## Synthetic multimodal samples

`multimodal-training-samples.png` は `qsa/mm_data.py` が生成する OCR・表・図形カウントの 8 例です。seed・質問・回答・各ソース PNG の SHA256 は `multimodal-training-samples.json` にあります。

```sh
uv run --locked --extra train python docs/assets/build_multimodal_samples.py
```

## Regenerate

Python 3.11+ に Matplotlib と Noto Sans CJK JP が必要です。プロジェクトの仮想環境を作らずに済むよう、`uv run --no-project --with matplotlib` で実行します。

```sh
uv run --no-project --with matplotlib python scripts/render_flowchart.py \
  docs/assets/qsa-overview.scene.json --out /tmp/qsa-overview --palette jade-violet
```

`--out` の場所に PNG / SVG と `.validation.json` が出ます。図を差し替えるときは PNG / SVG だけを `docs/assets/` へコピーしてください。

renderer は次を検査し、問題があれば書き出さずに終了します。

- 文字同士の重なり、ノードからの文字はみ出し、canvas 外への流出
- 矢印と他ノード・文字の交差

Scene JSON の主なキー:

| キー | 意味 |
|---|---|
| `canvas` | `[幅, 高さ]` |
| `nodes[].icon` | `lock db gear shield image chip flow check warn scale arrow layers` |
| `nodes[].role` | `neutral local global memory modulation`（配色を決める） |
| `nodes[].emphasis` | 濃い塗り + 白文字 |
| `edges[].from_side` / `to_side` | `top right bottom left`。省略時は位置から決める |
| `edges[].waypoints` | 折れ点を明示して検証を通す |
| `frames` / `captions` / `lines` | 背景帯、見出し、注記 |

アーキテクチャ図（`architecture/`）は原研究リポジトリの `docs/figures/qsa-architecture/build.py` で再生成します。PNG は 3600×2820 です。

図の事実の出典は `qsa-indexer-forge` revision `eb0fd3f`（アーキテクチャ図）と各 scene。詳細と評価条件は各図を参照している Markdown 文書に記載しています。
