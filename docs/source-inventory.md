# 元ディレクトリ・関連フォルダの構造

## パスの訂正

ユーザー指定の `~/projects/27b-qsa` は存在しませんでした。関連文書から正本を特定した結果、実体は `/home/ryu/OpenResearch/27b-qsa` です。`/home/ryu/projects/QSA27B` は学習リポジトリではなく、24GB 向け派生モデルの運用メモでした。

## 正本リポジトリ

`/home/ryu/OpenResearch/27b-qsa`、調査時 HEAD は `99ba95c`（`codex/mm2-multimodal-cache`）。Git 追跡対象は 684 ファイルです。

| パス | 追跡ファイル数 | 役割 |
|---|---:|---|
| `qsa/` | 12 | 現在の参照実装・export。最終学習コードは別 branch にある |
| `configs/` | 6 | 学習・評価・native 実行の設定 |
| `patches/` | 4 | mask 参照版、疎 kernel 版、MM2 cache 版の独立 patch |
| `scripts/` | 57 | 変換、fixture、検証、bench、配備 |
| `tests/` | 3 | QSA / HF / export のテスト |
| `docs/` | 84 | 計画、実装、評価、復元、公開、図 |
| `benchmarks/` | 87 | context、V100 Q4、拒否実験などの設計と結果 |
| `results/` | 387 | 集計 JSON、軽量な証拠、監査結果 |
| `handoff/` | 9 | MM Indexer の引継ぎと復元用 archive |
| `release/` | 29 | 公開用 model card、metadata、検証 |
| `models/` | 追跡外 | 旧 text-only HF export。実体は約 52GB |
| `.scratch/` | 追跡外 | 約419MB。仮想環境、一時 checkout、native 実験・公開作業の中間物 |
| `.cache/hf/` | 追跡外 | 約52GB の HF cache。`models/` と hardlink 等で重複計上の可能性 |
| `.archify/` | 追跡外 | 約2.3MB。アーキテクチャ図のレビュー中間物 |
| `.pytest_cache/` | 追跡外 | テスト cache |
| `release/engine-fork/` | 入れ子 Git | 専用 llama.cpp の実 checkout。外側の Git では追跡しない |
| `sakura-pagoda.html` | 未追跡・無関係 | 他作業の生成物。公開対象から除外 |

`results/` には層別 `positions.i32` / `selected.i32`、NLL、capture、検証 JSON を含みます。`benchmarks/` には固定 prompt と再実行 runner があります。大きな GGUF、生ログ、SATA 原本は Git へ入れない運用です。

## 学習コードが存在した branch

現在の共有 branch だけを見ると学習 loop が見つかりません。次の履歴が必要です。

| branch / commit | 内容 |
|---|---|
| `orx/p0-dense-baseline-oracle-capture*` | dense teacher、oracle capture、P0 |
| `orx/p1-indexer-distillation-k-2048-r-4` | 小型 overfit、8k/16k Indexer 蒸留 |
| `be2face` / `orx/p2-16k-k-2048-retrieval-hf-persistence` | step100 から 500 step を継続し step600 を保存 |
| `orx/mm0-*` | 既存 step600 のマルチモーダル診断 |
| `orx/mm1-lr-1e-4`, `orx/mm1-lr-3e-4` | 画像・テキスト混合の継続蒸留 |
| `d75da83` / `orx/mm2-16k-k-2048-300-step` | 最終 16k・300 step の学習コードと設定 |
| `codex/qsa-multimodal-export`, `codex/mm2-multimodal-cache` | HF export、native MM2、cache、公開 |
| `codex/text-inference-speed` | 重みを固定した速度開発 |

この公開リポジトリは `d75da83` の学習 tree を土台に、現行の export / cache runtime と検証を統合しています。学習時に必要な `logit_scale=1/sqrt(128)` と FP32 Indexer を失わないよう、現行 branch の古い学習側 `indexer.py` で上書きしていません。

## OpenResearch の実行記録

| パス | 役割 |
|---|---|
| `/home/ryu/.local/share/openresearch/orx.db` | run、実コマンド、backend、commit、時刻の正本 |
| `run-logs/<run-id>.log` | 学習の TRAIN / RESULT、wall time、評価 |
| `source-snapshots/<digest>.tar` | 各 run へ投入したソース |
| `worktrees/ac365d39-.../` | OpenResearch 実験ごとの worktree |
| `/home/ryu/output/27b-qsa-20261103/` | step100 / step600 の重みと再構成サマリ |

最終 text run は `57dd1e1d-54c5-4a62-95ab-b9fb47341056`、最終 MM2 run は `9dd0ce84-f7f2-46a6-92bb-33a915fc7ae8`。どちらも Modal A100 80GB です。

## 関連する作業ディレクトリ

| パス | 役割 |
|---|---|
| `/home/ryu/projects/QSA27B/` | IQ3_S + DFlash2、RTX 3090 / V100 の起動・bench・unit |
| `/home/ryu/projects/qsa-refusal-20261106/` | plain / dense / QSA の拒否挙動 3-arm 実験。学習ではない |
| `/home/ryu/OpenResearch/27b-qsa-quant/` | 量子化ナレッジベースの器。main 作業 tree は空で、実体は OpenResearch worktree |
| `/home/ryu/output/qsa-dir-map-20261106/` | 既存のディレクトリ構成図と検証記録 |
| `/home/ryu/output/qsa-keyvisual-20261106/`, `qsa-teaser-20261105/` | QSA の公開用 visual 作業。学習コードではない |
| `/home/ryu/projects/qsa-indexer-forge/` | この公開用リポジトリ |
| `/mnt/data2/qsa27b/`（z170） | GGUF、engine、実ログ、大きな原本。調査機からは未 mount |
| `/home/ryu/Backups/27b-qsa/` | 復元 core、研究・native の backup |

## 配布系統は2つ

1. 本流: `higa2515/Qwen3.8-27B-QSA-{BF16,GGUF}` + `kuraneko1/Qwen3.8-27B-QSA-llama.cpp`。
2. 24GB 派生: `pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF` + `pentaCoxian/Qwen-QSA-llama.cpp`。

IQ3_S / DFlash2 の派生を、Q8_0 / MM2 の本流と同じモデル・エンジンとして扱わないことが重要です。
