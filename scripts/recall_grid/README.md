# recall-grid と v2 の学習データ

`recall_grid.py` は、長い文脈から記録を正確に取り出せるかを測る生成器・採点器である（標準ライブラリだけで動く。`tokenizers` があれば token の位置が正確になる）。8 種類の課題（GUID、pointer の連鎖、レシピの材料と分量、最新の版、表の結合、コードの追跡、bindings）を、10 の深さ × 3 問ずつ、合計 240 問、本文の中に植える。どの問題にも、キーが 1 文字違う偽物の記録が付く。

indexer v2 の評価（rg-E・rg-V、seed 20261201・20261202、本文は PG-19 の test・validation の本）と、2 段の学習データは、どちらもこの生成器で作った。

| ファイル | 用途 |
|---|---|
| `recall_grid.py` | `gen`（問題を作る）、`check`（文書だけで解けるか確かめる）、`run`（OpenAI 互換のサーバに聞く）、`report`（集計と HTML） |
| `gen_train_16k.py` | 1 段目（300b）の学習データ。16k の文書、記録の先頭の token にラベル |
| `gen_mixup_64k.py` | 2 段目（mix64k）の学習データ。64k の文書、記録の全 token にラベル、取り違えの多い種類を重くする |
| `split_pg19_books.py` | PG-19 の parquet を 1 冊 1 ファイルに分ける（`gen_mixup_64k.py` の本文） |

手順と、できたデータの sha256 は [indexer v2 の学習を再現する](../../docs/training-v2.md) にある。

```sh
# 評価用の問題を作り、文書だけで解けることを確かめる
python scripts/recall_grid/recall_grid.py gen --out grid.jsonl.gz --tokenizer tokenizer.json --lengths 128000
python scripts/recall_grid/recall_grid.py check --data grid.jsonl.gz
# llama.cpp のサーバに聞く（thinking なし）
python scripts/recall_grid/recall_grid.py run --data grid.jsonl.gz --label mymodel \
  --base-url http://127.0.0.1:8080/v1 --llamacpp --no-think --max-tokens 65536
python scripts/recall_grid/recall_grid.py report --results results.jsonl --data grid.jsonl.gz --html report.html
```

- **本文で結果が変わる：** 既定の本文は、実行している Python の標準ライブラリのソースである。別の本文を使うときは `--filler`（ファイル、glob、ディレクトリ）で渡す。
- **比べ方：** 比べるときは、コマンドではなく、作ったデータのファイルそのものを共有する。
- **学習データを作るときの注意：** 評価に使う seed と本とは、別の seed・別の本を使う。
