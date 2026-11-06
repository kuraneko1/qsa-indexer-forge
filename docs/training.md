# Indexer 学習（Stage 1 distillation）

Qwen3.8-27B の full-attention 16 層に QSA（Qwen Sparse Attention）用の indexer を後付けし、
**backbone を凍結したまま** dense attention を教師にして蒸留する手順。実装は `qsa/train_indexer.py`
（テキスト）と `qsa/train_mm.py`（画像+テキスト）。

- 論文: [On the Design of Qwen3.8-Next Architecture](https://arxiv.org/abs/2608.30320), §2.1.2（Stage 1 / Eq. 17-20）
- 実行基盤の履歴: 27B を 1 枚に載せた A100 80GB。本リポジトリのコードは単一 GPU 前提（`device_map="cuda"`）。

## 0. 学習するもの / しないもの

| 区分 | 対象 |
|---|---|
| 学習する | 16 層ぶんの indexer のみ（**52.4M params / 80 tensors / FP32**） |
| 凍結する | 言語モデル本体（BF16）、vision encoder、projector、MTP 重み |
| 実装しない | Stage 2（backbone+indexer の sparse adaptation）、LoRA、MTP 用 indexer、vision 内部への QSA |

Stage 2（選択ブロック内だけで KL を取る 8,000 step / lr 2.5e-5 規模の学習）は**未実施**であり、
このリポジトリにも実装がない。ここで作るのは「Stage 1 の indexer」だけである。

## 1. 対象モデルと層

- ベース: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`（`configs/text-local.yaml:4-5`）
- 64 層のうち full attention は 0 始まりで **3, 7, 11, …, 63** の 16 層（`configs/mm2-integration.json` の `indexer.layers`）。
  残りは GDN（gated delta net）層で、`Qwen3_5Attention` 以外は触らない（`qsa/hf_qsa.py:118-147`）。
- **HF 参照 export** では画像・動画入力を dense に fallback する。MM 学習では画像埋め込みを含む系列について言語モデルの 16 層を蒸留するが、vision 内部は対象外。native MM2 では画像・動画も QSA 経路を使う。

## 2. indexer の構造

`qsa/indexer.py:47-144`。H=4 heads、head dim 128、部分 RoPE 64、ブロック圧縮 r=4、予算 K=2048 トークン（= 512 ブロック）。

1. `q_proj`: hidden → H\*D、`k_proj`: hidden → 1\*D（MQA。key は head 共有）
2. query / key に **zero-centered RMSNorm**（scale は `1 + weight`、初期値 0）→ `qsa/indexer.py:11-25`
3. key は **r=4 トークンの平均プール** → ブロック RMSNorm → **ブロック先頭位置の部分 RoPE**（`qsa/indexer.py:123-133`）
4. query は token 位置の部分 RoPE
5. `logits[b,h,t,n] = q·k` → head ごとに ReLU → head 和 → `* logit_scale`
   （`qsa/indexer.py:137`、`logit_scale = 1/sqrt(128)` は `qsa/hf_qsa.py:144` で注入）
6. ブロック因果マスク: 完全に観測済みのブロックだけを有効化（`qsa/indexer.py:139-143`）。
   末尾の不完全ブロックはスコア対象外（`qsa/indexer.py:107`）だが、推論時のマスクでは
   **自分の所属ブロックの末尾トークンを常に含める**（`qsa/sparse_attention.py:37-68`）。

注意: この `logit_scale` は順位を変えないが **KL の温度を変える**ので、蒸留結果は変わる。
過去の学習と条件を合わせる場合は必ずこの値を使う（変えるなら別実験として記録する）。

## 3. 教師分布（dense の attention をブロック化）

`qsa/teacher.py:16-54`。

1. 各 full-attention 層の dense 計算から Q/K を得る（GQA は head 展開してから）
2. query を 512 token ずつ chunk に切り、`softmax(q·k * scale)` を **head 和**（`amax` ではなく sum）
3. r=4 の完全ブロックごとに **max pool**（`teacher.py:48`）
4. 完全観測ブロックだけ残し、query ごとに **L1 正規化**（`teacher.py:51-53`）

`[B,H,T,T]` を作らないので 16k でも現実的に回る。教師は `fp16` で保持してメモリを節約する
（`qsa/hf_qsa.py:69` の `target.half()`）。

## 4. 損失

`qsa/losses.py:44-80`。

```
KL( teacher_block || softmax(indexer_scores) )    # valid ブロックのみ、teacher>0 の項のみ
```

- `valid` は「その query から見て完全に観測済みか」のマスク。`-inf` のスコアは softmax 前に落とす。
- 質量のない query（`target.sum(-1) == 0`）は平均から除外（`losses.py:77-80`）。
- マルチモーダル版は **per-query KL**（`losses.py:83-98`）に重みを掛けて平均する。
  画像ケースでは末尾 `question_span=96` トークン（質問・選択肢）に `question_weight=4.0`（`qsa/train_mm.py:176-182`）。
- Stage 2 用の `block_ids` 再正規化 KL も実装済みだが、**このリポジトリの学習では未使用**（`losses.py:60-67`）。

## 5. 1 step の流れ（テキスト）

`qsa/train_indexer.py:61-112`。

```
for each step:
  ids = 次の 16,384 token ブロック（batch=1）
  with no_grad: dense forward            # teacher と各層の入力 hidden / RoPE を保存
  for layer in 16 層:
      scores = indexer(hidden, cos, sin)  # fp32 で再計算（backbone の graph は使わない）
      loss   = KL(teacher, scores)
      loss.backward()                     # 層ごとに逆伝播（勾配は蓄積）
  clip_grad_norm_(indexer params, 1.0)
  optimizer.step(); optimizer.zero_grad()
```

- 最適化: `AdamW(lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)`（`train_indexer.py:216-217`）
- batch size は **1 固定**（indexer が `cos[0]`/`sin[0]` を使うため。`train_indexer.py:82`）
- 16k × 1 step が学習の最小単位。勾配蓄積やマルチ GPU は未実装。

## 6. データ

### テキスト
- `HuggingFaceFW/fineweb-edu` / `sample-10BT` を **streaming** で読み、文書を EOS で連結して
  `seq_len` ごとに切る（`qsa/data.py:11-46`）。ブロックは文書境界をまたぐ。
- revision は `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` に固定（`configs/text-local.yaml:21`、`qsa/data.py:29-30`）。
- 評価用ブロックは `skip_docs` をずらして取る（学習 0 / 評価 50,000。`configs/text-local.yaml:23-25`）。

### 画像（マルチモーダル）
- 画像は**ローカル生成の合成 3 種**で、外部データセットに依存しない（`qsa/mm_data.py:16,32-65`）。
  - `code-ocr`: 5 桁のアクセスコードを描画
  - `table-cell`: 2×2 の数値表
  - `shape-count`: 赤い円の個数
- seed から決定的に生成し、`GENERATOR_VERSION="mm0-synth-v1"` と各 PNG の SHA256 を記録（`mm_data.py:16,26-29,109-112`）。
- 1 step は「テキスト」か「画像+テキスト」のどちらか（`mm_ratio=0.5` で抽選。`qsa/train_mm.py:136-145`）。
  画像は詰め物テキストの **72% 位置**に置き、画像トークンが K ブロックを超える状況（N>K）を作る（`mm_data.py:147-153`）。
- 詰め物は fineweb から取り、`filler_pool=4` 本を用意して使い回す（`train_mm.py:33-46,113-117`）。

## 7. 実行

### 7.1 環境

```sh
cd qsa-indexer-forge
uv venv --python 3.12
uv pip install -e ".[train]"        # torch>=2.6, transformers==5.22.1, datasets, pillow, torchvision ...
uv run --locked --extra dev --extra train --extra export python -m pytest -q   # CPU テスト一式（CI と同じ）
```

- `transformers==5.22.1` は固定。`Qwen3_5ForConditionalGeneration` と
  `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention` に依存している（`qsa/hf_qsa.py:117,170`）。
  版が違うと import 時点で落ちる。
- `qsa/compat.py` は古い brotli/httpx2 を持つコンテナ向けの shim。`qsa/p0_capture.py:23-25` で自動適用される。
- GPU は **80GB 級 1 枚**を想定（BF16 の 27B が約 54GB、16k の活性が乗る）。
  コードは `device_map="cuda"` なので分割配置はしない。
- 初回は HF から 18 shard（約 54GB）をダウンロードする。ディスクに同等の空きが要る。

### 7.2 プロファイル

| config | stage | 開始点 | 出力 | 用途 |
|---|---|---|---|---|
| `configs/text-local.yaml` | `distill` | **ゼロから**（`step_offset: 0`） | `qsa_indexers*.pt` をローカル | 公開再現用のテキスト indexer を自分の環境で作る |
| `configs/multimodal-local.yaml` | `mm1` | 上の `qsa_indexers.pt`（`indexer_path`） | ローカル | テキスト→マルチモーダルへ継続 |
| `configs/multimodal-continue-mm2.yaml` | `mm1` | 公開 BF16 repo の `indexers/mm2_16k/qsa_indexers.pt` | ローカル | 公開済み MM2 重みから追加学習 |

3 つとも `push_to_hub: false` なので **HF トークン不要**。保存は常にカレントディレクトリ。
`save_every` ごとに `qsa_indexers_step{N}.pt`、最後に `qsa_indexers.pt` と `p1_results.json` / `mm1_results.json`。

### 7.3 コマンド

```sh
# テキスト indexer（600 step × 16,384 token ≒ 9.83M token）
python -m qsa.p0_capture --config configs/text-local.yaml

# その出力からマルチモーダル継続（300 step × 16,384 token）
python -m qsa.p0_capture --config configs/multimodal-local.yaml

# 公開 MM2 重み（higa2515/Qwen3.8-27B-QSA-BF16 @ d61281f4）から追加学習
python -m qsa.p0_capture --config configs/multimodal-continue-mm2.yaml
```

`stage` による分岐は `qsa/p0_capture.py:99-113`（`mm1` → `train_mm.run_mm1`、`mm0` → `mm0.run_mm0`、
`distill` → `train_indexer.run_distillation`、それ以外は P0 の dense capture）。

標準出力は 1 step ごとに
`TRAIN {"step":..,"loss":..,"capture":..,"teacher_entropy":..,"score_std":..,"seconds":..}`、
最後に `RESULT {...}` が 1 行出る。この 2 種類だけを保存すれば比較できる。

### 7.4 事前診断（任意・推奨）

学習前に dense のブロック捕捉率の上限を見たい場合は P0 を使う。P0 用のキー
（`budgets`, `n_seqs`, `ppl_len`, `ppl_seqs`）を持つ config を自前で用意する:

```yaml
# 例: configs/p0.yaml（このリポジトリには同梱していない）
model: Qwen/Qwen3.8-27B
model_revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
seq_len: 16384
n_seqs: 6
block_size: 4
budgets: [512, 1024, 2048, 4096]
dataset: HuggingFaceFW/fineweb-edu
dataset_revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
subset: sample-10BT
skip_docs: 0
ppl_len: 4096
ppl_seqs: 4
ppl_skip_docs: 10000
```

`python -m qsa.p0_capture --config configs/p0.yaml` で `p0_results.json` が出る。

## 8. 実測リソース（履歴）

最終学習は Modal の **A100-SXM4-80GB（79GiB）** で実行した。所要時間は run ログの実測。

| 内容 | step × 長 | 学習ループ | wall（起動〜終了） | 1 step |
|---|---|---|---|---|
| テキスト P2 / step600 | 500 × 16,384（新規 8.19M、累計 9.83M tok） | 6,351.6 s | 118 分 | 約 12.7 s |
| マルチモーダル MM2 300 step | 300 × 16,384 | 66.1 分 | 68.7 分 | 約 13.2 s |
| MM1 pilot（4k） | 300 × 4,096 | 720.1 s | 12.3 分 | 約 2.4 s |
| P0 dense capture | 6 × 16,384 | — | 9.0 分 | capture 62.1 s |

- 16k の 1 step が 12〜13 秒なので、**600 step で約 2 時間、300 step で約 1.1 時間**が目安。
  これに 54GB の重み取得とロード（数分）が加わる。
- peak メモリは当時のログに残っていない（未記録）。OOM する場合は `seq_len` を 8,192 に落とす
  （履歴でも 8k の 600 step が 60 分で回っている）。

## 9. 履歴の結果（参考値）

| run | 内容 | final_loss | final_capture | dense PPL | sparse PPL | 差分 |
|---|---|---|---|---|---|---|
| テキスト 600 step @16k | 公開 step600 の元 | 0.2619 | 0.9414 | 7.0730 | 7.0718 | -0.017% |
| MM2 300 step @16k | 公開 mm2_16k の元 | 0.3115 | 0.9353 | 7.0730 | 7.0714 | -0.022% |

数値の出典は `configs/text-step600-results.json`、`configs/mm2-training-results.json`（run `9dd0ce84` の RESULT から転記）と `configs/mm2-integration.json`。
**これらは 1 seed・小標本の測定であり、品質同等の証明ではない**（評価の限界は `docs/evaluation.md`）。

実験名・ブランチ名は実設定と一致しないことがある。履歴の
`orx/p1-indexer-distillation-k-2048-r-4` は名前が `k-2048` だが、実際に流した config は
**`seq_len: 8192` / `budget: 1024`**（8k の 12.5%）であり、K=2048 は P2 からである。
条件は名前ではなく **config の値**で判断すること。

## 10. 再現性の限界（必読）

1. **bit 単位の再現はできない。** 当時の学習はデータを streaming で読み、入力トークン列・data cursor を
   保存していない。checkpoint は weights-only で、optimizer / scheduler / RNG 状態を含まない。
   同じ設定・同じ step 数でも重みは一致しない。
2. `configs/text-local.yaml` は **ゼロからの 600 step** であり、履歴（step100 から warm start して 500 step）の
   再実行ではない。step 数と学習量を合わせた「同規模の再学習」である。
3. `configs/multimodal-continue-mm2.yaml` は**公開済み MM2 重みからの追加学習**であり、MM2 の再現ではない。
4. データの revision は固定したが、`datasets` の streaming 実装・HF 側の parquet 分割・
   `transformers` の tokenizer 実装が変わればトークン列は変わり得る。**原 run の依存は freeze されていない**。公開 repo の `uv.lock` は今回検証した環境を固定するが、原 run と同じ依存集合を証明するものではない。
5. GPU が違えばカーネル差で数値は変わる（同じ A100 でも浮動小数の順序に依存する箇所がある）。
6. Stage 2（backbone sparse adaptation）と、Stage 2 用の `block_ids` 制限 KL は未実施・未使用。

## 11. 既知の問題・注意

- マルチモーダルの開始点は 2 通り。`indexer_path`（ローカル `.pt`）が優先され、無ければ
  `indexer_repo`/`indexer_file`/`indexer_revision` を HF から取得する（`qsa/train_mm.py:94-107`）。
  `indexer_sha256` を書いておくと不一致で即停止する（`train_mm.py:101-103`）。**HF の既存重みを取得する設定では SHA を必ず書く**。`text-local.yaml` で自分が生成した直後のローカル checkpoint には不要。
- 実行前に設定の整合を確認できる: `pytest tests/test_configs.py`（公開 config が
  `push_to_hub: false`、revision 40 桁、`budget/block_size` 固定であることを検査する）。
- `configs/historical/` は**当時の設定の原本**であり、そのまま実行しない。
  `push_to_hub: true` で `higa2515/qwen38-27b-qsa-indexers` へ書き込みに行き、この repo は現在 **401（非公開/権限なし）**。
  参照用に残しているだけで、実行すると認証エラーで停止する（または意図しない送信を試みる）。
- **マルチモーダルの pilot は「完走しているが failed 記録」**である。履歴の MM1 2 arm（lr=1e-4 / 3e-4）は
  300 step を完走して `RESULT` 行と checkpoint を出した後、**プロセス終了時の shutdown crash（abort）**で
  exit code が非 0 になり、run の status は `failed` になった。成果物とログは有効だが、
  「failed = 学習が壊れた」と読み替えないこと。同様の abort は完走後に起き得るので、
  `RESULT` 行と `qsa_indexers.pt` の存在で成否を判断する。
- 学習・評価は FineWeb-Edu（ODC-By）と Qwen3.8-27B（Apache-2.0）の条件に従う。詳細は `NOTICE.md`。
- `*.pt` / `*.safetensors` / `*.gguf` は `.gitignore` 済み。生成物をコミットしない。

## 12. 学習後（変換）

- HF 形式へ: `python -m qsa.export_model --base <27B snapshot> --indexers qsa_indexers.pt --output <dir>`
  （`qsa/export_model.py:184-192`）。backbone 18 shard は無改変、indexer を別 shard に FP32 で追加し、
  `conversion_manifest.json` に来歴を書く。
- 量子化済み GGUF へ移植: `scripts/transplant-qsa-gguf.py`（80 indexer tensors を FP32 のまま追加、
  元の 866 tensors を byte 単位で保持）。詳しくは `docs/native-runtime-history.md` と
  `docs/native-multimodal-history.md`。
