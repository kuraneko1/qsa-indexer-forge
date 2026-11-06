# 評価

学習した indexer（と QSA 経路）をどう測るか。実装は `qsa/train_indexer.py`（テキスト）、
`qsa/train_mm.py`（マルチモーダル）、`qsa/mm0.py`（学習前診断）。数値の意味と限界を先に読むこと。

- 評価は **1 seed・小標本**。統計検定も信頼区間も実装していない。差が出ないことを「同等」と書かない。
- capture（捕捉率）は**教師分布に対する指標**であり、下流タスクの品質ではない。
- PPL・capture から応答挙動（拒否・冗長化など）を推測しない。挙動は挙動として別に測る。

## 1. 3 層の評価

| 層 | いつ | 中身 | 実装 |
|---|---|---|---|
| 学習中 | 毎 step | capture、teacher_entropy、score_std | `qsa/train_indexer.py:89-112`、`qsa/train_mm.py:176-197` |
| 学習後（自動） | run 末尾 | dense vs sparse PPL、passkey retrieval、MM capture、MM 4 択 | `train_indexer.py:222-322`、`train_mm.py:209-271` |
| 学習前（任意） | 事前 | dense の oracle capture（予算スイープ）、MM の dense 整合と capture | `qsa/p0_capture.py`、`qsa/mm0.py` |

## 2. capture（ブロック捕捉率）

```
capture = Σ_{query} Σ_{b ∈ 学習器が選んだ top-kb} teacher_b
          ─────────────────────────────────────────────
          Σ_{query} Σ_{b ∈ 完全観測ブロック}      teacher_b
```

- `kb = ceil(K/r) = 512`（K=2048, r=4）。テキストは `select_blocks` の top-k（`qsa/sparse_attention.py:11-34`）、
  MM は `torch.topk` を直接使う（`qsa/train_mm.py:49-54`）。
- teacher はすでに完全観測ブロック上で L1 正規化されているので、そのまま質量比になる。
- **N<=K の query は除外**する（可視ブロックが kb 以下だと全選択になり 100% になるため）。
  MM の実装は除外数を `queries_excluded_N_le_K` として出力する（`qsa/mm0.py:203-211`）。
- **oracle** は「教師自身の global top-k」であり、セル別の上限ではない。そのため学習側の capture が
  oracle をわずかに上回ることがある（`configs/mm2-integration.json` の limits にも記載）。
- capture は `seq_len` と K の比に強く依存する。**4k で 0.99、16k で 0.94** のような値をそのまま比較しない
  （4k では K=2048 が全体の 50%、16k では 12.5%）。

## 3. dense vs sparse PPL

`qsa/train_indexer.py:29-48`。

- 同一ブロック列を 2 回 forward し、`budget=0`（dense attention）と `budget=K`（QSA）で比較する。
- full-attention 層だけを差し替え、GDN 層は共通。logits は 512 token ずつ chunk して cross-entropy。
- 既定は評価 2 系列 × 16,384 token（`eval_seqs: 2`、`eval_skip_docs: 50000`。`configs/text-local.yaml:24-25`）。
  実測は **32,766 token**（(16,384-1)×2）。
- 出力: `dense_ppl`、`sparse_ppl`、`ppl_delta_pct = 100*(sparse/dense-1)`。

## 4. passkey retrieval

`qsa/retrieval.py:25-68`。

- fineweb の文書を詰め物にし、5 桁の passkey を中央に挿入、4 択（A-D）で末尾 1 位置の logits を比較。
- 長さは 8,192 / 16,384 / 32,768 を指定（実際のトークン数は 8,133 / 16,325 / 32,709 などになる）。
- 各長 8 問、1 問 1 forward（cache 非対応の patched attention でも動く）。
- dense と sparse の両方で正答率を出し、**満点なら識別力がない**（履歴は両方 1.0）。

## 5. マルチモーダル capture（4 象限）

`qsa/mm_capture.py:16-93`。

- ブロックを modality（image / text）に分け、query も modality で分けて
  **text→text / text→image / image→text / image→image** の 4 セルで capture を出す。
- さらに末尾 `question_span=96` トークンだけの **question 領域** capture も出す（`train_mm.py:234-269`）。
- セルごとに queries / mass も出すので、「画像を見る query が増えただけ」なのか
  「text query の選択が良くなった」のかを切り分けられる。
- mass は「そのセルにどれだけ教師質量があるか」。text→image は質量が小さく、比だけ見ると過大評価しやすい。

## 6. マルチモーダル 4 択 QA（合成）

`qsa/train_mm.py:229-271`。

- `make_specs(seed+1000)` で評価用の合成ケースを作り（学習と別 seed）、dense と sparse の
  最終位置 logits で A-D を選ばせ、正解と比較する。
- `eval_cases: 2` が既定。**n=2 で精度を語らない**。履歴では dense 0/2・sparse 0/2（MM2）、
  MM0 で dense 4/8・sparse 3/8、MM0b で 4/8・5/8 であり、いずれも結論を出せる規模ではない。
- 画像の内容（OCR の数字・表の値・図形の数）は生成時に保持しているので、必要なら人手で確認できる。

## 7. MM0（学習前の診断）

`qsa/mm0.py:35-212`。凍結した indexer のまま、dense との数値整合と capture を測る。

- dense 経路と sparse 経路の最終 logits の最大差（`parity_max_abs`）と top-1 一致数（`parity_top1`）を出す。
  履歴の診断 run では 4,096/6,400 token 級で **max_abs 0.0、top1 8/8**。
- 短いケース（`n_short`）と長いケース（`n_long`、`long_tokens`）を混ぜ、画像は `image_position: late`。
- **indexer は HF からしか読まない**（`mm0.py:89-90`）。ローカルの `.pt` を使いたい場合は
  `indexer_repo`/`indexer_file`/`indexer_revision` を指すか、当該箇所を `indexer_path` 対応に変える。

実行例（P0 と同じく `stage: mm0` の config を用意する）:

```yaml
stage: mm0
model: Qwen/Qwen3.8-27B
model_revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
image_token_id: 248056
indexer_repo: higa2515/Qwen3.8-27B-QSA-BF16
indexer_revision: d61281f4626729e4cf138476a704c7fedc08f891
indexer_file: indexers/qsa_indexers.pt
indexer_sha256: e44193e5ef832b6220c8355fff92ac8f07d8c13538f584d2b6231b5dd2f4bc51
block_size: 4
budget: 2048
dataset: HuggingFaceFW/fineweb-edu
dataset_revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
subset: sample-10BT
skip_docs: 0
long_tokens: 6400
n_short: 2
n_long: 6
question_span: 96
seed: 0
```

`python -m qsa.p0_capture --config <この config>` で `mm0_results.json` と `RESULT` 行が出る。

## 8. 実行コマンド（学習と同時に走る）

```sh
# テキスト: 学習後に dense/sparse PPL と retrieval まで自動
python -m qsa.p0_capture --config configs/text-local.yaml

# マルチモーダル: 学習後に PPL・MM capture・4択QA まで自動
python -m qsa.p0_capture --config configs/multimodal-local.yaml
python -m qsa.p0_capture --config configs/multimodal-continue-mm2.yaml
```

評価だけをやり直したい場合は、同じ checkpoint から短い step 数で再開するのではなく
**学習 run の出力（`p1_results.json` / `mm1_results.json`）とログを保存して比較する**。
過去の run では `p1_results.json` がアップロード順の不具合で残らなかった事例がある
（`configs/text-step600-results.json` の note）。ログの `RESULT` 行を必ず残すこと。

## 9. 履歴の実測値

### 9.1 事前（P0, A100 80GB, 16k×6 系列）

| 予算 K | oracle capture |
|---|---|
| 512 | 0.8864 |
| 1024 | 0.9300 |
| 2048 | 0.9622 |
| 4096 | 0.9845 |

dense PPL 7.7873（4,096 token × 4 系列 = 16,380 token）。K=2048 の上限は約 96%。

### 9.2 テキスト 600 step

| 設定 | final_loss | capture | dense PPL | sparse PPL | 差分 | retrieval |
|---|---|---|---|---|---|---|
| 16k（公開 step600） | 0.2619 | 0.9414 | 7.0730 | 7.0718 | -0.017% | dense/sparse とも 1.0（8k/16k/32k） |
| 8k / K=1,024（比較用） | 0.2618 | 0.9367 | 7.1977 | 7.1949 | -0.039% | 未測定 |

出典: `configs/text-step600-results.json`。8k の行は **予算 K=1024**（8k の 12.5%）。16k の行は K=2048（12.5%）。
同じ「600 step」でも capture の意味が違うので、2 行を直接比較しない。

### 9.3 マルチモーダル

| run | step×長 | final_loss | capture | PPL 差分 | question 領域 capture（learned / oracle） |
|---|---|---|---|---|---|
| MM1 pilot | 300×4k | 0.2163 | 0.9926 | — | text→text 0.9725 / 0.9808、text→image 0.9943 / 0.9953 |
| MM2（公開） | 300×16k | 0.3115 | 0.9353 | -0.022% | text→text 0.8958 / 0.9179、text→image 0.9912 / 0.9923 |

MM0b の 4 象限（16k 級、N<=K を 209,504 query 除外）:
text→text 0.9671 / 0.9793、text→image 0.9271 / 0.9575、image→text 0.9671 / 0.9815、image→image 0.9988 / 0.9999（learned / oracle）。

### 9.4 外部データ（このリポジトリには同梱しない）

DocVQA（`nielsr/docvqa_1200_examples` rev `dc77ab0c…`、CC BY-NC-SA 4.0・非商用）を
同一 checkpoint（`mm2_16k/qsa_indexers.pt`, sha256 `e44193e5…`）で 2 通りに測った履歴がある。
**同じデータでも評価設計で数字が全く変わる**ので、両方を残す。

| 評価設計 | n | dense | sparse | top-1 一致 | NLL | competent |
|---|---|---|---|---|---|---|
| 自由回答（文字列一致） | 50 | **0.24** | **0.22** | 0.98 | 未計測 | false |
| 構成 4 択（A-D の logits）+ teacher-forced NLL | 50 | **1.00** | **1.00** | 1.00 | 0.7562 / 0.7639 | true |

- 1 行目は自由回答の文字列一致。27B の DocVQA 自由回答としては低く、**評価系が機能していない**
  （`competent: false`）と記録された。
- 2 行目は選択肢を A-D に限定し、最終位置の logits で選ばせたもの。上限に張り付いており、
  こちらも**識別力がない**（dense と sparse が同じ 1.0）。
- 入力長は 1,456〜6,878 token で **50 件中 49 件が K=2048 超**。予算比が固定の比較ではない。
- どちらも公式 DocVQA スコアではなく、データは同梱しない。**0.24 → 1.00 の変化をモデル改善と読まない**
  （変わったのは評価設計であって重みではない）。

### 9.5 量子化 GGUF 上での長文監査（当時の別スクリプト・本リポジトリ外）

学習済み indexer を Q4/Q5/Q6 の GGUF に移植した状態で、実文章の約 64k 入力を測った履歴がある。
12,288 next-token 予測、capture は 64k 地点。

| モデル | 全地点 PPL 差 | 64k 全体 | 64k 文章のみ | 64k capture | 同予算 oracle |
|---|---|---|---|---|---|
| UD-Q4_K_M | -1.412% | -0.397% | **+2.444%** | 73.82% | 82.93% |
| UD-Q5_K_M | -0.472% | +0.065% | **+1.972%** | 73.76% | 82.91% |
| UD-Q6_K | -0.346% | -0.073% | **+1.756%** | 73.89% | 83.04% |

- 「全地点平均」はコード窓の改善に引っ張られて負になる。**実文章の 64k では 3 つとも PPL が上がる**。
- 64k では選択率が約 **3.15%**（16k の 12.5% とは予算比が違う）。16k の capture と直接比較できない。
- oracle との差は約 9 ポイントあり、追加学習の余地は残る。ただし BF16 の同条件対照が無いため、
  この上昇を量子化起因とも indexer 起因とも断定できない。
- この監査スクリプト群は本リポジトリに含まれない（`docs/native-runtime-history.md` 系の記録のみ）。

## 10. 解釈の限界（必読）

1. **capture は「教師の質量を取りこぼしていないか」だけ**。教師自身が間違っていれば、よく一致しても品質は上がらない。
2. **PPL は 32,766 token の 1 標本**。信頼区間が無いので、±0.05% 程度の差を「同等」「改善」と断定できない。
   履歴の差分（-0.017% / -0.022%）は、この規模では分解能の外にあると考えるのが妥当。
3. **retrieval は天井**（4 択・8 問・両 arm 満点）。識別力がないので、この指標だけで長文能力を語らない。
4. **MM の 4 択は n=2〜8**。しかも合成画像（OCR・表・図形）だけで、実写真・文書・動画の分布を代表しない。
   履歴の MM2 は dense 0/2・sparse 0/2 で、設問自体が難しすぎる可能性がある。
5. **モデル間・条件間で capture を直接比較しない**。seq_len、K、r、層数、学習 token 数が違えば意味が変わる。
6. **量子化（GGUF）後の品質は別評価**。このリポジトリの学習は BF16 の HF モデル上で行われる。
   Q4/Q5/Q6/Q8 での indexer 監査は当時の別スクリプト群で実施しており、本リポジトリには含まない。
7. **応答挙動の評価は本リポジトリの範囲外**。拒否率・過少拒否・形式遵守などは別の実験系で測る
   （PPL や capture を代理指標にしない）。
8. Stage 2（backbone sparse adaptation）は未実施なので、**「QSA を入れて品質が保たれた」とは書けない**。
   ここで保証できるのは「Stage 1 indexer が dense 教師のブロック選択をどれだけ再現したか」まで。
9. **同じ重み・同じデータでも評価設計で結論が反転する。** DocVQA は自由回答で 0.24/0.22、
   構成 4 択に変えると 1.0/1.0 になった。片方だけ引用すると誤った印象を与える。必ず両方と設計を書く。
10. **コンテキスト長で結論が変わる。** 16k の PPL はほぼ横ばいでも、量子化 GGUF の実文章 64k では
   Q4/Q5/Q6 で +2.44% / +1.97% / +1.76% 上がった。16k の結果を長文の保証に使わない。
11. **MM2 の合成 4 択は dense も 0/2**。dense 側が解けていないので、dense と sparse を区別できていない
   （比較として不成立）。n を増やすか、設問を易しくするまで精度を語らない。

## 11. 報告テンプレ

| 項目 | 書く内容 |
|---|---|
| チェックポイント | SHA256、学習 step、seq_len、warm start の有無 |
| 学習条件 | lr、batch（=1）、token 数、データ revision、seed |
| capture | 値と、seq_len / K / 除外 query 数 |
| PPL | dense / sparse、token 数、評価ブロックの skip_docs |
| retrieval | 長さごとの正答数（満点なら「天井」と明記） |
| MM | 4 象限 + question 領域、QA は n を明記 |
| 実行環境 | GPU 名、ライブラリ版、所要時間 |
| 限界 | 上記 10 章のうち該当する項目 |
