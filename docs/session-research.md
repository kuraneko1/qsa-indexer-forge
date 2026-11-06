# 関連セッションの網羅調査

Indexer の学習経路と llama.cpp 対応を、コードだけでなく会話・実験・判断の記録から照合しました。公開するのは関連事実の要約であり、認証情報、無関係な会話本文、base64 payload は含めません。

## 保存系ごとの詳細

| 対象 | 走査範囲 | 詳細 |
|---|---|---|
| Codex | `~/.codex/sessions` の 248 rollout、archived 5 件、session index / SQLite / history | [Codex セッション全記録](codex-sessions.md) |
| OpenResearch | `orx.db`、27B-QSA の 6 chat + Quant 1 chat、15 experiments / 31 runs、63 run logs、source snapshots / worktrees | [OpenResearch の実験・会話・時系列](openresearch-research.md) |
| DeepSeek Harness | `~/.dsh` と `~/.dsh-latest` の 1,670 セッションを3 pass で全文走査 | [DSH セッション全記録](dsh-sessions.md) |

## 統合した時系列

| 日時 (JST) | 保存系 | 出来事 |
|---|---|---|
| 2026-09-04〜11-03 | Codex 前史 | Flash-Next の QSA / MTP、indexer の文脈走査、QSA kernel を調査 |
| 11-03 03:25 | Codex | 27B への適用可否、計算資源、コスト、評価計画をレビュー |
| 11-03 07:25 | OpenResearch / DSH | 27B-QSA project を開始。P0、参照実装、Modal A100 の実行 |
| 11-03 10:10 | OpenResearch | P1 8k・600 step。実設定は K=1024（branch 名の K=2048 とは異なる） |
| 11-03 13:37〜15:28 | OpenResearch | P2 16k・K=2048、step100 から 500 step を継続し step600 を保存 |
| 11-03 16:09〜 | Codex | native llama.cpp 移植、mask 参照版、疎 CUDA、MTP、公開の主実装 |
| 11-04 01:54 | OpenResearch / DSH | MM0/MM1 の handoff。step600 から画像・テキスト継続蒸留を設計 |
| 11-04 02:17〜05:33 | OpenResearch | MM0 / MM0b / MM1 / 最終 MM2 16k・300 step、QA / DocVQA 評価 |
| 11-04 03:38〜10:03 | Codex / DSH | MM2 native cache、M-RoPE、MTP 画像 embedding、prefix 再利用、公開 |
| 11-04〜05 | DSH | DSH 自体を QSA モデルで駆動し、長考・compaction・実運用を検証 |
| 11-05 | Codex / DSH | UD-Q4/Q5/Q6 への Indexer 移植、64k 監査、IQ3_S / DFlash2 派生の配備 |
| 11-06 | DSH | plain / dense / QSA の短文・長文 3-arm 評価、品質比較 |
| 11-06 13:19〜 | DSH | この公開リポジトリへ学習・native・session 記録を統合 |

## セッション照合で分かった重要な補正

1. **P1 の名前と実設定は違う**: branch 名は `k-2048` だが、成功した 8k run の config は `budget: 1024`。K=2048 の学習は P2 から。
2. **MM1 の `failed` は学習失敗とは限らない**: 両 arm とも 300 step と RESULT / checkpoint 出力後、shutdown abort で終了した。
3. **DocVQA は評価設計で結果が反転**: 自由回答 0.24 / 0.22 から、構成 4 択では 1.0 / 1.0。後者だけを品質証明に使わない。
4. **最終 MM2 の公開取得先は BF16 repo**: 旧 `qwen38-27b-qsa-indexers` は現在 401。`Qwen3.8-27B-QSA-BF16` の `indexers/qsa_indexers.pt` が現存する公開経路。
5. **engine は複数系統**: `6209237 + patch` / 公開 `e697cf2c` の本流と、IQ3_S 向け `pentaCoxian/Qwen-QSA-llama.cpp` を分ける。
6. **実装の主 Codex session だけでは全学習を再現できない**: 初期 commit を作った rollout は Codex 保存側に無く、OpenResearch / DSH 側の記録が必要だった。

## 網羅性の境界

- z170-desktop 内部の `~/.codex`、`~/.dsh`、SATA 上の実ログは今回のローカル走査の対象外です。正本に記録された絶対パスと hash を来歴として保持しています。
- 11-03 の初期コミット群に対応する Codex rollout は見つかりませんでした。OpenResearch の DB と DSH session がその証拠です。
- DSH の短い空 session、QSA モデルで駆動しただけの無関係作業、Codex の同名 cwd の無関係作業は、関連実装の証拠と混同しないよう別扱いです。
- 検索は `QSA` の短い文字列を base64 / 暗号化 payload に多数検出したため、語境界と強い別名で再検索しています。詳細なクエリと除外例は各詳細文書に記録しています。
