# DSH セッション調査（27b-qsa / QSA / Indexer / llama.cpp）

- 調査日: 2026-11-06（JST）
- 対象: `/home/ryu/OpenResearch/27b-qsa` とその周辺（QSA indexer の学習、GGUF 化、llama.cpp 対応、z170 運用、拒否率検証）
- 方針: 読み取り専任。原本・既存ファイルは変更せず、本ドキュメントのみ新規作成
- 走査規模: `~/.dsh/sessions` 1662 ファイル（690 MB, zstd 圧縮）+ `~/.dsh-latest/sessions` 8 ファイル = 1670 ファイル

## 1. 実保存場所

| 種別 | パス | 実体 |
|---|---|---|
| セッション本体 | `~/.dsh/sessions/<cwdエンコード>/<session-id>/session.v4.jsonl.zstd` | zstd 圧縮 JSONL（1 行 1 イベント）。一部 `v3`。cwd は `/`→`-`、非 ASCII は `~XXXX`（UTF-16 コード単位） |
| メタ情報 | `~/.dsh/storages/session_projcache/sessions/<id>.json` | 1678 件。`identity.createdAt` / `cwd` / `rows.title` |
| 検索 DB | `~/.dsh/storages/session-index.sqlite` | FTS5 スキーマのみ存在し **行数 0**。全文検索には使えない |
| UI 状態 | `~/.dsh/storages/workspace.json` | 57 KB。ワークスペースとタブの状態 |
| 別ホーム | `~/.dsh-latest/sessions/...` | 8 件のみ（`/home/ryu/ドキュメント/deepseek-harness/default-workspace`）。QSA モデルの試験運用に使われた |

保存先ディレクトリ名の例:

- `/home/ryu/OpenResearch/27b-qsa` → `--home-ryu-OpenResearch-27b-qsa--`
- `/home/ryu/projects/QSA27B` → `--home-ryu-projects-QSA27B--`
- `/home/ryu/projects/qsa-refusal-20261106` → `--home-ryu-projects-qsa-refusal-20261106--`
- `/home/ryu/.local/share/openresearch/worktrees/ac365d39-.../chat_76f1417a-...` → `--home-ryu-.local-share-openresearch-worktrees-ac365d39-...-chat_76f1417a-...--`

## 2. 検索方法

1. `find ~/.dsh/sessions ~/.dsh-latest/sessions -name 'session.v*.jsonl.zstd'` で全ファイル列挙
2. 各ファイルを `zstd -dc -q <file>` でストリーム展開し、Python でバイト列に対し正規表現カウント（JSON パースより高速で、巨大ファイルでも破綻しない）
3. 3 パス構成
   - 第 1 パス: 素の部分一致（`qsa` `qwen3.8` `pentacoxian` `llama.cpp` `indexer`）— ノイズが多い
   - 第 2 パス: 境界ガード付き（前後が英数・`+/=_-` の場合は除外）＋前後に 140/200 文字の文脈を保存
   - 第 3 パス: 第 1・2 パスで得た別名（`qsa27b` `qsa_indexers` `qwen3.8-27b-qsa` `mm2_16k` `Qwen-QSA-llama.cpp` `QwenSparseAttention` `step600`）で再検索
4. `session_projcache` の JSON と突き合わせて `createdAt` / `cwd` / `title` を付与（日時はすべて JST、`createdAt` 基準）

## 3. クエリ一覧

| パス | クエリ |
|---|---|
| 1 | `27b-qsa` `qsa` `qwen3.8` `qwen3-8` `pentacoxian` `penta` `llama.cpp` `llamacpp` `indexer` `iq3_s` `gsq` `rco` `qsa-indexer` |
| 2 | `\bqsa\b` `27b-qsa` `qsa[-_ ]?indexer` `pentaCoxian` `llama\.?cpp` `qwen3\.8` `qwen3\.8-27b` `qwopus` `q38` |
| 3 | `qsa27b` `qsa_indexers` `qwen3\.8-27b-qsa` `mm2_16k` `qsa-mm2` `gsq[-_]rco` `Qwen-QSA-llama\.cpp` `QwenSparseAttention` `sparse attention` `step ?600` |

誤ヒットの除外:

- **base64 断片**: 画像・バイナリの base64 に `qsa` が偶然一致する（初回 287 件中 6 割超）。前後が `[A-Za-z0-9+/=_-]` なら不一致として除外。
- **スキルカタログ由来の定型文**: `qwen3.8`（4 回）と `GSQ-RCO`（2 回）は全セッションのシステムプロンプトに含まれるため、出現回数だけでは判定に使わない。
- **`indexer` 単独**: llama.cpp のコード indexer と QSA indexer が混在。`qsa` / `27b-qsa` / `qsa_indexers` との共起でのみ採用。
- **`penta` / `pentacoxian` 単独**: DSH のアイコン作者クレジット（PENTACoXIAN）として無関係セッションにも出る。QSA 文脈との共起で判定。
- **`rco`**: `record` 等の部分一致が多く、QSA 文脈では使わない。

## 4. ヒット数と採否

| 段階 | 件数 |
|---|---|
| いずれかの別名にヒット | 978 |
| `qsa` / `27b-qsa` / `pentaCoxian` 系 | 179 |
| 強い別名（`qsa27b` `qsa_indexers` `qwen3.8-27b-qsa` `mm2_16k` `Qwen-QSA-llama.cpp`） | 261 |
| 関連として採用（本体） | 27（初回走査 26 + 後続の公開監査 1） |
| QSA モデルで駆動されたセッション | 21（本体表とは別枠・重複なし） |
| 空・無関係 | 5（うち 1 件は第 6 表に再掲） |

## 5. 関連セッション（本体・時系列）

| 日時 (JST) | セッション ID | cwd / 保存先 | 役割・内容 |
|---|---|---|---|
| 2026-10-21 05:51 | `session-39729697-1ef2-4f90-a5ae-ea3b5fec08be` | `/home/ryu/projects/v100` | 前史。`pentaCoxian/Qwen-V100-llama.cpp` の起動と prefill/decode 監視。`:8091` 監視の追加 |
| 2026-10-26 01:00 | `session-abfc79f2-240b-4713-bbad-0dcc9c8a1624` | `/home/ryu/projects/z170/Nanbeige` | 前史。z170 運用リポジトリ取得、量子化比較（GSQ 系）、Qwen3.8-27B QSA IQ3_S（`:8091`）の運用記述 |
| 2026-11-03 07:25 | `7a56c632-6269-4995-a7af-69a65d67495c` | orx worktree `ac365d39…/chat_76f1417a` | `QwenSparseAttention` を Qwen3.8-27B へ適用する研究の着手。modal A100（$2.50/h）の費用判断と停止の決定 |
| 2026-11-03 09:39 | `0ea43929-36bd-45bf-9d8e-00563d608eff` | 同 worktree | `qsa_indexers` / `.pt` の実体確認、蒸留成果物の所在整理 |
| 2026-11-03 11:57 | `532ae120-e39a-46ae-a69a-81c77ec08877` | 同 worktree | `/workspace/repo/qsa_indexers.pt` を 8 MiB ずつ base64 で 4 並列に吸い出し |
| 2026-11-04 01:54 | `d9975041-150d-4234-9283-60918c56ae43` | orx `ac365d39…/chat_58a27ab3` | 既存 text-only indexer を step600（`warm100/qsa_indexers_step600.pt`）から画像＋テキストで継続蒸留する計画。KL 蒸留、block max pool、L1 正規化、MM0/MM1 の段階設計 |
| 2026-11-04 02:10 | `7519335f-b8f7-42d8-a832-b79e13dec94c` | 同 worktree | 上記計画の補足・確認（短セッション） |
| 2026-11-04 03:38 | `session-833e467c-78df-42c9-9526-dbf5e0bd36f7` | `/home/ryu/OpenResearch/27b-qsa` | 起動コマンドの確定。`ggml-runtime-v5`、`LLAMA_QSA_RESERVE_BLK/BLK_REBUILD/MASKED_PREFILL_MAX`、MTP の `KV_ONLY/PAD/N_UBATCH`、自動再起動 off、port と sm tensor 版 |
| 2026-11-04 04:10 | `session-27d01aad-8e43-4d78-9ec2-5b795ff6577a` | 同 | 周辺。コンパクション設定の見直し（QSA 本体ではなく DSH 側の調整）。他セッションの ID 参照あり |
| 2026-11-04 04:27 | `session-b9f8f021-1953-4685-8e66-cf279e811708` | 同 | ディレクトリ解説。MM2 重み＋Q8_0 GGUF＋現行エンジンの関係整理 |
| 2026-11-04 05:26 | `eb519967-6970-4ab0-acff-60ef44ebf891` | orx worktree `9433bc86…/chat_5f3c5a84` | 量子化調査。Q8_0 はほぼ無損失、**indexer は F32 固定が必須**、同一 base の GSQ-RCO GGUF から非 indexer 866 tensors を移植可能 |
| 2026-11-04 08:24 | `e8e0a665-59b5-4846-9f30-c6de47157bad` | orx `chat_76f1417a` | indexer を量子化しない Q8_0 の再作成、画像 prefix キャッシュが再利用されない問題の修正 |
| 2026-11-04 14:07 | `session-0eb2da4e-1a7d-4a8e-a41d-b0e491fb6bcb` | `/home/ryu/OpenResearch/27b-qsa` | 公開資料整備。HF/GitHub README の日英化、**公開リポジトリ名から MM2 を除去**、pentaCoxian 名義の追記、QSA 説明と速度比較図、Contributors 整理 |
| 2026-11-05 01:49 | `session-09a4364d-ad14-4a11-8ce9-bdb45c1f7804` | 同 | ディレクトリ把握から **拒否率比較の計画**（huihui dense vs QSA、logprob のみ読むプローブ、H1=indexer 不適合仮説）、git 整理、GitHub private 公開 |
| 2026-11-05 01:50 | `session-f7ab68b2-de2e-44fd-b2bc-d3b4521e5d0d` | `/home/ryu/projects/QSA27B` | `pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF` を IQ3_S で z170 へ導入、エンジンは `pentaCoxian/Qwen-QSA-llama.cpp`、v100+RTX3090 構成へ入れ替え |
| 2026-11-06 01:41 | `session-4402af2b-8941-48f2-be18-c1747c277b0e` | sandbox-workspace | 周辺。z170 運用（`z170-ops`）で QSA IQ3_S `:8091` の起動・health を扱う |
| 2026-11-06 07:32 | `session-73b20318-363c-457b-811e-882332327e1d` | `/home/ryu/projects/qsa-refusal-20261106` | handoff から実装継続。huihui abliterated Q4KM QSA を `:8192` に立て、dense/QSA の arm 比較を実行、黒プロンプトセットを作成 |
| 2026-11-06 12:04 | `99f652c5-afd8-496c-85f3-3a46aa1e3eef` | 同 | 10.3k 資料に対する 2 arm 応答の品質判定（第 1 便） |
| 2026-11-06 12:16 | `0f55e285-eade-4c05-839a-227422fb35ff` | 同 | 同判定の第 2 便（長文 black セット） |
| 2026-11-06 13:19 | `session-2152a0db-f7eb-4b83-9081-d1a680ffaa43` | `/home/ryu/projects/qsa-indexer-forge` | 現行オーケストレーションの親。Indexer 学習と llama.cpp 対応の調査・公開リポジトリ化を統括 |
| 2026-11-06 13:23 | `304c61ef-6628-4ee6-9c56-55f3cd03a650` | 同 | 子。`~/projects/27b-qsa` 不在の指摘と、正本 `/home/ryu/OpenResearch/27b-qsa` の学習方法特定 |
| 2026-11-06 13:23 | `59eac534-0c2c-4ecc-96dd-eb85ca6b3f6a` | 同 | 子。`QSA27B/DEPLOY-IQ3S.md` から公開情報・上流リポジトリ・コミットを特定 |
| 2026-11-06 13:23 | `bdf4ec16-e924-4c7a-892a-2e7280528563` | 同 | 子。QSA/Indexer を llama.cpp で動かすための全変更（パッチ・CUDA/CPU カーネル・GGUF）を追跡 |
| 2026-11-06 13:24 | `b4509692-2187-4f74-b7d6-6ad69b813622` | 同 | 子。OpenResearch 側のセッション記録・handoff・実験ログの横断 |
| 2026-11-06 13:24 | `222c004f-6dfd-4c9f-8703-fe0601e76771` | 同 | 子。本調査（DSH セッション網羅）を担当 |
| 2026-11-06 13:24 | `c06de259-4cfa-4326-82f2-eefec935c2f7` | 同 | 子。Codex セッション側の網羅調査 |
| 2026-11-06 | `b08039ce-4825-4ba8-9944-78c453c46edb` | 同 | 子。公開直前の第三者視点レビュー（初回走査後に追加） |

## 6. QSA モデルで駆動されたセッション（`qsa-qwen/qwen3.8-27b-qsa-text`）

`model/selection` の実測で 21 件。DSH 自体を QSA モデルで動かした運用検証で、長考・コンパクション・モデル切替の挙動が議論対象になっている。

| 日時 (JST) | セッション ID | cwd / 保存先 | 備考 |
|---|---|---|---|
| 2026-11-04 03:54 | `session-f76308cd-8185-4f7b-a05a-cde6dd113a2c` | `/home/ryu/OpenResearch/27b-qsa` | 挨拶のみ（24 行） |
| 2026-11-04 03:55 | `session-5202fac9-58c1-4a8e-a0f7-24b9ec800640` | 同 | ディレクトリ調査と QSA 適用 27B の既存事例調査。voxel 生成の寄り道を含む |
| 2026-11-04 04:23 | `session-4172a82b-3dad-4dc0-bad2-74b37626738d` | 同 | voxel 塔の生成のみ（内容は無関係、駆動モデルは QSA） |
| 2026-11-04 04:51 | `session-786eb5c4-6c07-47a3-b15a-eab57a3c294c` | sandbox-workspace | セッション運用の検証 |
| 2026-11-04 05:14 | `session-f78034a5-8dfb-458d-8c03-67d53bd81d80` | 同 | 同 |
| 2026-11-04 05:16 | `session-e8c65f54-d2c9-42ff-b87b-a9aa1a043767` | 同 | 思考ターンの再開・予算切れの観測（他セッションから参照） |
| 2026-11-04 06:00 | `session-962be809-cc18-480a-b900-654eba8ff6b1` | 同 | 同 |
| 2026-11-04 06:00 | `session-c4630ba1-3ff3-456b-a9d3-7eea7e68ec3c` | 同 | 同（1.5 MB） |
| 2026-11-04 08:05 | `session-694b80c3-7f7c-4d0e-91f6-aeebf5b72630` | 同 | 同 |
| 2026-11-04 16:31 | `session-2d63ea08-fa64-496b-94dd-6edc96ec623a` | 同 | 最長（3.1 MB）。長時間運用の挙動 |
| 2026-11-05 02:41 | `session-3bd5fdaa-540c-4426-bec7-c233afc1407a` | 同 | 同 |
| 2026-11-05 03:45 | `session-9f92f692-0436-4d3f-a0cf-ce631f0a9794` | 同 | 同 |
| 2026-11-05 03:57 | `session-586bd579-83c7-4dbe-9641-69e6c21549f2` | 同 | 同 |
| 2026-11-05 06:09 | `session-8bca07ff-044c-4a84-906f-bc31c4d5c31d` | `~/.dsh-latest`（default-workspace） | 別ホームでの試験 |
| 2026-11-05 06:10 | `session-53779b5d-1920-48d0-a2a7-65b674c69081` | sandbox-workspace | 同 |
| 2026-11-05 06:27 | `session-6af7632a-cb2d-4833-b746-23defc3fbd26` | `~/.dsh-latest` | 別ホーム |
| 2026-11-05 06:28 | `session-3548c1b7-2c7d-4248-9d1a-31604e68715c` | `~/.dsh-latest` | 別ホーム |
| 2026-11-05 06:30 | `session-90436ac2-caa4-4532-b051-30f9b456dda2` | `~/.dsh-latest` | 別ホーム |
| 2026-11-05 08:25 | `session-c0556bbb-5f94-4ece-a27b-b2ecf549c98a` | `~/.dsh-latest` | 自己紹介・基本動作の確認 |
| 2026-11-06 01:50 | `session-b552efc5-e811-491a-ba38-ffa9173b95e8` | `~/.dsh-latest` | 短い確認 |
| 2026-11-06 01:54 | `session-3a529791-f492-4584-8a8a-98e8f285056b` | `~/.dsh-latest` | 同 |

## 7. 空・無関係セッション

| 日時 (JST) | セッション ID | 保存先 | 判定 |
|---|---|---|---|
| 2026-11-05 02:40 | `session-b69bbea0-9408-45f4-9163-e9a4863aeb00` | `/home/ryu/projects/QSA27B` | 5 行で終了（内容なし） |
| 2026-11-06 06:36 | `session-919c8ec5-5af5-418c-8418-4de9ae6ebbfa` | `/home/ryu/OpenResearch/27b-qsa` | 6 行で終了 |
| 2026-11-06 08:39 | `session-eef42c68-aa08-4709-b418-4c785ff5f515` | `qsa-refusal-20261106` | 36 行、試し入力のみ |
| 2026-11-06 11:02 | `session-b5b3b184-a3f8-4464-9e11-1c3df81e800a` | 同 | 5 行で終了 |
| 2026-11-04 04:23 | `session-4172a82b-3dad-4dc0-bad2-74b37626738d` | `/home/ryu/OpenResearch/27b-qsa` | 内容は voxel 生成で無関係（第 6 表に再掲） |

## 8. 重要な決定

- **学習**: text-only indexer（step600）を初期値に、画像＋テキストで継続蒸留する方針。蒸留は teacher attention の head softmax 和 → block max pool → L1 正規化を KL で合わせ、backbone・vision・projector・MTP は凍結。計画段階の制約は「modal の A100 は高コストなので使い続けない」。
- **量子化**: Q8_0 を基準に維持。indexer は F32 固定（converter が F32 でも `llama-quantize` は 2D weight を落とすため、明示的に守る）。同一 base の GSQ-RCO 公開 GGUF とは非 indexer 866 tensors が名前・形とも一致し、型割当のコピーや移植が現実的。
- **公開**: HF リポジトリ名から `MM2` を除去（重み・revision・本番設定は不変）。GitHub は `pentaCoxian/Qwen-V100-llama.cpp` の fork、既定ブランチ `codex/qsa27b-mm2`。上流へ PR は出さない。
- **運用**: z170 の `:8091` に IQ3_S（GSQ-RCO）を置き、エンジンは `pentaCoxian/Qwen-QSA-llama.cpp`。v100+RTX3090 の混在構成。
- **検証**: 拒否率比較は huihui の dense と QSA の 2 arm。生成せず logprob のみ読むプローブを先に置き、H1（indexer が無検閲 backbone に不適合）を検証する設計。2026-11-06 の実行では短い 46 プロンプトで 3 arm が byte-identical という結果が記録されている。

## 9. 関連コミット・revision

| 種別 | 値 |
|---|---|
| 学習 commit（正本 repo） | `d75da83`（MM2 16k: SHA-checked warm start plus text PPL and MM QA evaluation, 2026-11-04） |
| 学習 run ID | `9dd0ce84-f7f2-46a6-92bb-33a915fc7ae8` |
| エンジン土台 | `6209237d`（`release/engine-fork` に存在） |
| 公開エンジン HEAD | `e697cf2c`（branch `codex/qsa27b-mm2`） |
| HF: GGUF | `b8bfeee6`（`higa2515/Qwen3.8-27B-QSA-GGUF`） |
| HF: BF16 | `d61281f4`（`higa2515/Qwen3.8-27B-QSA-BF16`） |
| HF: indexer repo | `b51b8d15`（`higa2515/qwen38-27b-qsa-indexers`, `mm2_16k/qsa_indexers.pt`） |
| indexer SHA256 | `e44193e5ef832b6220c8355fff92ac8f07d8c13538f584d2b6231b5dd2f4bc51` |
| 全差分 patch | `patches/llama-qsa27b-mm2-cache.patch`（SHA256 `68f1c9e0…`） |

## 10. 網羅性の限界

- 走査対象は `~/.dsh` と `~/.dsh-latest` のセッションのみ。他ホスト・他ユーザー・Codex 側（`~/.codex/sessions`）・OpenResearch のチャット DB は別系統で、本ドキュメントの対象外。
- セッションは zstd 圧縮 JSONL で、外部の全文検索インデックスは存在しない（`session-index.sqlite` は 0 行）。文字列一致ベースのため、画像・音声・圧縮バイナリの中身は判定できない。
- 別名の網羅は「初ヒットから得た語」に依存する。未知の内部呼称が別名として使われていた場合、未検出の可能性がある。
- `penta` / `indexer` / `qwen3.8` などは一般語・定型文と衝突するため、単独ヒットは採用していない（誤検出より漏れを許容する側に倒した箇所がある）。
- 日時はセッション作成時刻（`createdAt`）基準。最終更新は mtime であり、長期継続セッションでは数日ずれる。
- 内容の要約は会話ログの一部（ユーザー発話・重要な assistant 発言）に基づく。全 step の完全な再構成ではない。
- 認証情報・API キー・個人ファイルの内容は意図的に読まず、記載もしない。
