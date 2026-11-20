# indexer v2 の学習を再現する（300b → mix64k）

公開版の v2 indexer は、v1（MM2 16k）から 2 段の追加学習で作った。このページは、その 2 段をこの repo のコードで回す手順である。

- **経緯と評価：** [公開版 indexer から v2 まで](research-v2.md)。
- **2 段目の詳細：** [取り違え重点の 64k 追加学習](training-64k-mixup.md)。
- **土台の蒸留：** [Indexer 学習ガイド](training.md)（教師分布、KL、学習ループ）。

| 段 | 設定 | 初期値 | 系列 | データ | 計算（公開版の実測） |
|---|---|---|---|---|---|
| 1 段目（300b） | [`configs/recall-16k-300b.yaml`](../configs/recall-16k-300b.yaml) | v1（BF16 repo のタグ `v1`） | 16,384 × 300 step | 合成 240 step ＋ FineWeb-Edu 60 step | A100 80GB、約 75 分 |
| 2 段目（mix64k = v2） | [`configs/mixup-64k.yaml`](../configs/mixup-64k.yaml) | 1 段目の結果 | 65,536 × 300 step | 合成 300 問 | A100 80GB、5.26 時間、メモリの山 78.7 GB |

実際に回したときの設定は、[`configs/historical/recall-16k-300b.yaml`](../configs/historical/recall-16k-300b.yaml) と [`configs/historical/mixup-64k.yaml`](../configs/historical/mixup-64k.yaml) に残してある（そのままでは動かさない）。

## 1. 学習データを作る

生成器は [`scripts/recall_grid/`](../scripts/recall_grid/README.md) にある。どちらも CPU だけで動き、tokenizer の `tokenizer.json` が要る（公開版では BF16 repo `aedae2c` のもの、sha256 `0997f410…`）。

### 1.1 1 段目：16k の記録探し（600 問）

```sh
uv run --locked --extra train python scripts/recall_grid/gen_train_16k.py \
  --out data/recall-train-16k.jsonl.gz --docs 200 --doc-tokens 16384 --density 0.6 \
  --tokenizer tokenizer.json --seed 1001
```

- **本文：** recall_grid の既定（実行している Python の標準ライブラリのソース）。
- **ラベル：** 答えに要る記録の先頭の token（`answer_pos`）と、偽物の記録の先頭の token（`decoy_pos`）。
- **実際に使う問題：** 合成のデータは先頭から順に読むので、1 段目で使うのは先頭の約 240 問である。

### 1.2 2 段目：64k の取り違え重点（300 問）

本文は PG-19 の **train** の本。公開版では `emozilla/pg19` の 1 分割だけを使った。

```sh
# PG-19 train の 1 分割（data/train-00000-of-00023-5263dc4323e881d2.parquet）を、1 冊 1 ファイルに分ける
uv run --locked --extra train --with pyarrow python scripts/recall_grid/split_pg19_books.py \
  --out books train-00000-of-00023-5263dc4323e881d2.parquet
uv run --locked --extra train python scripts/recall_grid/gen_mixup_64k.py \
  --out data/mixup-64k.jsonl.gz --runs 25 --q-per-doc 3 --length 64000 \
  --tokenizer tokenizer.json --filler-dir books --seed 2026111700
```

- **分割：** 1,244 冊が 1,185 ファイルになる（同じ題名の本は上書きされる）。
- **規模：** 25 回 × 64k の文書 4 本 × 3 問 = 300 問。どの問題も 1 回だけ使う。
- **ラベル：** 答えに要る記録と偽物の記録の、全 token。
- **種類の重み：** revision 4、pointer 3、code 2、guid 2、ほかは 1。

### 1.3 できたデータの確かめ方

gzip のヘッダには時刻が入るので、展開した中身の sha256 で比べる（`zcat FILE | sha256sum`）。

| ファイル | 行数 | 展開した中身の sha256 |
|---|---:|---|
| `recall-train-16k.jsonl.gz` | 600 | `aa08541dfbe6a8f615e82ad64d2eaffc4e9949e20450e3d768359ffcd36650b1` |
| `mixup-64k.jsonl.gz` | 300 | `a94d687c83657456d1e847657ccfe561921f23abdf9109b0a8ca42f837125d8a` |

2026-11-20 に、この repo の生成器（Python 3.14.6、tokenizers 0.23.3）で作り直した結果が、学習に使ったデータと一致した。PG-19 の分割も、本の名前と中身がすべて一致した。

## 2. 1 段目（300b）を回す

```sh
uv run --locked --extra train python -m qsa.p0_capture --config configs/recall-16k-300b.yaml
```

- **初期値：** v1 の indexer を、BF16 repo のタグ `v1` の commit（`aedae2c`）から取り、sha256 `e44193e5…` を確かめてから使う。
  - main の `indexers/qsa_indexers.pt` は、2026-11-20 から v2 である。
- **損失：** 教師との KL ＋ 0.3 ×（1 − 問題文の token での正解ブロックの質量）＋ 0.1 × relu(2.0 −（正解の最高点 − 偽物の最高点））。
- **データの混ぜ方：** `mix_fineweb: 0.2` で、step の番号を 10 で割った余りが 8・9 のときに FineWeb-Edu を使う。
  - 公開版を回したときの設定は 0.3 だったが、当時の `combine_iterators` の書き方のせいで、実際は 0.2 になっていた。いまの関数と 0.2 で、同じ順番になる。
- **保存：** 50 step ごとに保存し、最後に `qsa_indexers.pt` を作る。公開版の結果は sha256 `5b760e46…`。

## 3. 2 段目（mix64k）を回す

```sh
cp qsa_indexers.pt qsa_indexers_300b.pt   # 1 段目の結果
QSA_QCHUNK=1024 QSA_TEACHER_QCHUNK=256 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --locked --extra train python -m qsa.p0_capture --config configs/mixup-64k.yaml
```

- **GPU：** 80GB の GPU が 1 枚要る（A100 80GB でメモリの山 78.7 GB）。
- **1 段目からの変更：**
  - 系列長を 64k にした。
  - μ を 0.3 にした。
  - `restrict_kl` で、選んだ上位 512 ブロックの中だけで KL を取る。
  - `kl_rows: 4096` で、教師と KL を、問題文の全 token と、step ごとにランダムな 4,096 の文書位置だけで計算する。
- **保存：** `step_offset: 300` なので、チェックポイントの名前は step325〜600 になる。公開版の step600 は sha256 `794cbfed…`。
- **1 step の時間：** 約 63 秒（A100 80GB）。`kl_rows` を外すと約 145 秒になる。

## 4. 増えた設定の項目

| 項目 | 意味 |
|---|---|
| `local_data` | 記録探しの合成データ（jsonl.gz）。1 行が「文書＋質問」の 1 系列 |
| `local_instruction` | 質問の後ろに付ける指示（`{}` に答えの形式が入る） |
| `mix_fineweb` | FineWeb-Edu の割合（10 分の 1 単位） |
| `lambda_answer` | 正解ブロックの質量の項の重み（λ） |
| `mu_margin`、`margin` | 偽物との差の項の重み（μ）と、差の目標 |
| `restrict_kl` | 選んだ上位ブロックの中だけで、教師を正規化し直して KL を取る（論文の式 20） |
| `kl_rows` | 教師と KL を取る文書位置の数（0 は全位置）。乱数は `seed × 1000003 ＋ step の通し番号` |
| `loss_chunk` | 損失と逆伝播を、この token 数ずつ区切って計算する |
| `spill_capture`、`offload_embed` | 捕らえた教師などをホストの RAM に置く、入力の埋め込みを CPU に置く |
| `indexer_repo`、`indexer_revision`、`indexer_file`、`indexer_sha256` | Hub から取る初期値を固定し、sha256 を確かめる |
| `init_local`、`init_sha256` | 手元のファイルを初期値にする（`init_sha256` があれば確かめる） |
| `local_eval_data`、`local_eval_n` | 学習の後に、別のデータで正解ブロックの順位を測る（任意） |
| 環境変数 `QSA_QCHUNK` | sparse の注意を、問題の位置この数ずつに区切る（64k で 15.6 GiB の確保を避ける） |
| 環境変数 `QSA_TEACHER_QCHUNK` | 教師の計算の区切り（既定 512） |
| 環境変数 `QSA_GDN_CHUNK`、`QSA_PROFILE` | GDN の chunk 幅を縮める、1 step の時間の内訳を出す |

区切って計算しても値が変わらないことは、`tests/test_v2_training.py` で確かめている（CPU、float64）。

## 5. 結果を使う

- **Hugging Face 形式：** `python -m qsa.export_model --indexers qsa_indexers.pt ...` で、元の BF16 に indexer を足した形にできる（[README](../README.md#hugging-face-形式へ-export-する)）。
- **GGUF：** その HF 形式から、エンジンの変換器で作る（[llama.cpp 対応ガイド](llama-cpp.md)）。
  - 公開版の v2 GGUF は、v1 の GGUF の indexer の 80 テンソル（F32）だけを書き換えて作った。その書き換えの道具は、この repo には入っていない。

## 6. 同じ重みになるか

- **同じ GPU・同じ環境：** 2 段目は、同じ Colab の A100 で 2 回回し、最初の 3 step の loss がビット単位で一致した。step325 のチェックポイントの sha256 も一致した。
- **違う GPU・違うライブラリ：** ビット単位の一致は保証しない。公開版は torch 2.15.0+cu132、transformers 5.22.1 で回した。
- **1 段目の評価：** 公開版の 1 段目は、学習の後に別のデータ（seed 2002 の 60 問のうち先頭 20 問）で正解ブロックの順位を測っていた（`local_eval_data`）。この測定は重みに影響しないので、公開の設定からは外した。
