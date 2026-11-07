# ドキュメント目次

## 学習から実行までの読み順

1. [Indexer 学習ガイド](training.md) — 対象層、教師分布、KL、学習設定、実行コマンド、所要時間。
2. [llama.cpp 対応ガイド](llama-cpp.md) — HF / GGUF 変換、model load、QSA graph、疎 CUDA、MTP、cache。
3. [評価と限界](evaluation.md) — dense 対照、capture、PPL、retrieval、実生成、未検証範囲。
4. [再現チェックリスト](reproduction.md) — 段階ごとの手順、合格条件、つまずきやすい点、再現できない範囲。

## アーキテクチャ

[変換前 / 変換後の対比](architecture.md) — 64 層（GDN 48 + 全注意 16）+ MTP の構成を保ったまま、16 層に Indexer を追加した位置関係、KV と MTP の扱い。

| 版 | PNG | 編集用 SVG | フォント非依存 SVG | PDF |
|---|---|---|---|---|
| 日本語 | [PNG](assets/architecture/qsa-architecture-ja.png) | [SVG](assets/architecture/qsa-architecture-ja.svg) | [Outlined](assets/architecture/qsa-architecture-ja-outlined.svg) | [PDF](assets/architecture/qsa-architecture-ja.pdf) |
| English | [PNG](assets/architecture/qsa-architecture-en.png) | [SVG](assets/architecture/qsa-architecture-en.svg) | [Outlined](assets/architecture/qsa-architecture-en-outlined.svg) | [PDF](assets/architecture/qsa-architecture-en.pdf) |

## 図解

| 図 | 内容 | 編集用ソース |
|---|---|---|
| [全体像](assets/qsa-overview.png) | 凍結 → Indexer 蒸留 → 形式変換 → 専用 llama.cpp → 評価の 5 段階 | [SVG](assets/qsa-overview.svg) / [scene](assets/qsa-overview.scene.json) |
| [学習から llama.cpp 実行まで](assets/qsa-training-flow-mid.png) | 入力・凍結、テキスト蒸留、MM 継続、HF / GGUF、専用 runtime、評価 | [SVG](assets/qsa-training-flow-mid.svg) / [scene](assets/qsa-training-flow-mid.scene.json) |
| [llama.cpp QSA 実行パイプライン](assets/llama-cpp-pipeline.png) | convert → load → graph → inference、dense / fallback、検証 | [SVG](assets/llama-cpp-pipeline.svg) / [scene](assets/llama-cpp-pipeline.scene.json) |
| [dense / QSA の統制評価](assets/evaluation-flow.png) | 同一条件での分岐、同じ指標での比較、結果と未測定範囲の保存 | [SVG](assets/evaluation-flow.svg) / [scene](assets/evaluation-flow.scene.json) |
| [MM 学習の合成画像8例](assets/multimodal-training-samples.png) | OCR・数値表・図形カウントを seed 付きで表示 | [manifest](assets/multimodal-training-samples.json) / [生成 script](assets/build_multimodal_samples.py) |

工程図とアーキテクチャ図はベクター描画で、アイコンもベクターです（合成画像の例だけは実画像のコラージュ）。工程図は **mid** 詳細度。実装を説明する図であり、新たな GPU 実測や品質保証の証拠ではありません。

## 再現性・来歴

- [来歴と固定 revision](provenance.md) — モデル、Indexer、engine、run ID、commit、SHA256。
- [元ディレクトリの構造](source-inventory.md) — 正本、関連作業場、OpenResearch の保存物、branch の対応。
- [GGUF utilities](../scripts/README.md) / [llama.cpp patches](../patches/README.md)。

## 関連セッションと研究記録

- [セッション調査の入口](session-research.md)
- [Codex セッション](codex-sessions.md)
- [OpenResearch の実験・会話](openresearch-research.md)
- [DeepSeek Harness セッション](dsh-sessions.md)

## 当時の native 記録

- [FlashNext runtime の移植履歴](native-runtime-history.md)
- [MM2 multimodal / cache の移植履歴](native-multimodal-history.md)

これらは履歴写しです。現在の手順は本目次の「学習から実行までの読み順」を優先してください。