# 再現チェックリスト

第三者が、公開物だけでどこまで再現できるかを段階ごとに整理した文書です。手順の背景は
[Indexer 学習](training.md) と [llama.cpp 対応](llama-cpp.md)、数値の解釈は [評価と限界](evaluation.md) を参照してください。

- 最終確認: 2026-11-07（JST）
- 確認したこと: CPU テスト、公開 config の妥当性、HF の固定 revision の到達性、3 patch の適用可否、各 CLI の起動
- **確認していないこと**: 27B の学習・CUDA build・27B 推論の再実行。本リポジトリには 80GB 級 GPU が無く、
  本調査は文書とコードの静的確認、CPU で回る範囲の実行に限っています

## 1. どこまで再現できるか

| 段階 | 何が手に入る | 再現度 |
|---|---|---|
| L0 実装の検証（CPU） | テスト 33 件、Indexer の式・教師分布・KL・変換・cache の検証 | **そのまま再現可** |
| L1 事前診断（P0 / MM0） | dense の oracle capture、dense PPL、凍結 indexer の parity | 再現可（A100 80GB） |
| L2 テキスト蒸留 | 0 から 600 step の Indexer、dense/sparse PPL、retrieval | **同規模の再学習**（履歴の bit 一致ではない） |
| L3 マルチモーダル継続 | 画像＋テキストの 300 step、4 象限 capture、4 択 QA | 同上 |
| L4 HF export | backbone 無改変＋80 FP32 テンソルの追加 shard | 再現可（CPU で変換のみ） |
| L5 llama.cpp | 固定 engine、または base commit への patch 適用と build | 適用可・build 以降は GPU 環境依存 |
| L6 外部評価（DocVQA 等） | 当時の数値 | **再現不可**（データ非同梱・評価スクリプトは本リポジトリ外） |

蒸留の結果は、データを streaming で読み、入力トークン列・data cursor・optimizer / RNG 状態を保存していないため、
**同じ重みをビット単位で再生成することはできません**。同じ設定・同じ step 数の「同規模の再学習」です。

## 2. L0: GPU なしで確認する

```sh
git clone https://github.com/kuraneko1/qsa-indexer-forge
cd qsa-indexer-forge
uv run --locked --extra dev --extra train --extra export python -m pytest -q
```

33 件が通れば、Indexer の順伝播・教師分布・KL・block 選択・HF 変換・GGUF cache の実装は
この環境で動いています。個別に見るなら次が短い確認になります。

```sh
# 1 バッチに過学習できるか（蒸留ループの実装確認、GPU 不要・数秒）
uv run --locked --extra dev --extra train python -m pytest tests/test_overfit_tiny.py -q

# 公開 config の固定 revision / budget / block_size の検査
uv run --locked --extra dev --extra train python -m pytest tests/test_configs.py -q
```

## 3. 前提（L1 以降）

| 項目 | 要件 | 理由 |
|---|---|---|
| GPU | **80GB 級 1 枚**（A100 80GB で実績） | BF16 の 27B が約 54GB。`device_map="cuda"` で分割しない |
| ディスク | 60GB 以上 | 18 shard（約 54GB）＋ checkpoint |
| Python | 3.11 以上 | Transformers 5.22.1 の要求 |
| 依存 | `uv.lock` のとおり | `transformers==5.22.1` 固定。変えると import で落ちる |
| ネットワーク | 初回のみ | HF から重み・データを取得。`push_to_hub: false` なので認証不要 |

80GB に満たない場合は `seq_len` を 8,192 に落とす（履歴でも 8k の 600 step が 60 分で回っています）。
その場合 K=2048 は系列長の 25% になるため、**16k の capture と直接比較しないでください**。

## 4. L1: 事前診断（任意・推奨）

```sh
python -m qsa.p0_capture --config configs/p0.yaml      # dense のみ。p0_results.json
python -m qsa.p0_capture --config configs/mm0.yaml     # 公開 MM2 indexer の parity と capture
```

合格の見方:

- P0: `capture.overall` が K とともに単調増加し、K=2048 で 0.94〜0.97 程度（履歴は 0.9622）。
- MM0: `parity_max_abs` が 0（patch 前後で logits が一致）、`parity_top1` が全問一致。

## 5. L2 / L3: 蒸留

```sh
python -m qsa.p0_capture --config configs/text-local.yaml                # 600 step
python -m qsa.p0_capture --config configs/multimodal-local.yaml          # 上の出力から 300 step
python -m qsa.p0_capture --config configs/multimodal-continue-mm2.yaml   # 公開 MM2 から 300 step
```

保存物（すべてカレントディレクトリ）:

| ファイル | 内容 |
|---|---|
| `qsa_indexers.pt` | 最終 checkpoint（weights-only） |
| `qsa_indexers_step{N}.pt` | `save_every` ごとの中間 checkpoint |
| `p1_results.json` / `mm1_results.json` | 評価結果の JSON |
| 標準出力の `TRAIN` / `RESULT` 行 | step ごとの loss・capture、run 末尾の要約 |

合格条件は **`RESULT` 行が出て `qsa_indexers.pt` が存在すること**です。プロセス終了時の abort で
exit code が非 0 になっても、`RESULT` と checkpoint が残っていれば学習は完走しています（履歴の MM1 が該当）。

履歴の目安（16k・A100 80GB）: 1 step 12〜13 秒、600 step で約 2 時間、300 step で約 1.1 時間。
初回はこれに 54GB の取得とロードが加わります。

## 6. L4: export と検証

```sh
uv run --locked --extra export python -m qsa.export_model \
  --base /path/to/Qwen3.8-27B --indexers qsa_indexers.pt --output /path/to/Qwen3.8-27B-QSA
uv run --locked --extra export python -m qsa.validate_export /path/to/Qwen3.8-27B-QSA
```

- 元の 18 shard・tokenizer・processor・vision・MTP は無改変、Indexer だけが別 shard に FP32 で入ります。
- `validate_export` はモデルを instantiate せずテンソル配置だけを検査します。
- 出力先が既にある場合は上書きせず失敗します（途中生成物を消してから再実行）。

## 7. L5: llama.cpp

```sh
git clone https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp
cd Qwen3.8-27B-QSA-llama.cpp && git checkout e697cf2c72ecaa165de3714ea6ac3b00649ab274
```

- 変換・build・起動・dense 対照は [llama.cpp 対応](llama-cpp.md)。
- 別 base から始める場合は `6209237df53569f07429f3f79fc5354de5fe704f` に **patch を 1 つだけ**適用します。
  2026-11-07 時点で、3 patch は同梱の engine snapshot に対して `git apply --check` を通ることを確認済みです。
  重ねて適用しないでください。

## 8. つまずきやすい点

1. **P1 の名前と設定が違う**。branch 名は `k-2048` でも、実行したのは 8k / `budget: 1024`。
2. **公開 checkpoint のパス**。`indexers/qsa_indexers.pt` ではなく `indexers/mm2_16k/qsa_indexers.pt`（BF16 repo `d61281f4`）。
3. **`configs/historical/` は実行しない**。`push_to_hub: true` で、送信先は現在 401。
4. **capture は系列長と K の比で変わる**。4k の 0.99 と 16k の 0.94 を並べない。
5. **dense と sparse の PPL 差（±0.02%）は分解能の外**。32,766 token の 1 標本です。
6. **DocVQA のような外部評価は非同梱**。数値だけを引用しない。
7. **MM の 4 択は n=2〜8**。dense も 0/2 の履歴があり、比較として成立していません。

## 9. この文書の更新

手順や固定 revision を変えたら、この表と [来歴](provenance.md) を同時に更新してください。
「再現できた」と書くときは、実行した日・GPU・step 数・`RESULT` 行の値・checkpoint の SHA256 を添えてください。
