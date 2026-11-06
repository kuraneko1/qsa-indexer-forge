# 27B-QSA 関連 Codex セッション調査（読み取り専任で実施）

2026-11-06 時点で、このホストの Codex セッション保存領域を全件走査し、27B-QSA
（Qwen3.8-27B への Qwen Sparse Attention 適応、indexer 蒸留、llama.cpp 移植）に
関するセッションを特定・整理した記録。調査は読み取りのみで、既存ファイルは変更していない。

- 対象リポジトリ: `/home/ryu/OpenResearch/27b-qsa`
- 調査日: 2026-11-06
- 認証情報、無関係セッションの本文は収集・記載していない。

## 1. 検索範囲

| 種別 | 場所 | 規模 | 用途 |
|---|---|---:|---|
| rollout 本体 | `~/.codex/sessions` | 248 ファイル / 527 MB | 全文走査の主対象（2026-05〜2026-11） |
| archived rollout | `~/.codex/archived_sessions` | 5 ファイル / 1.4 MB | 同上（アーカイブ分） |
| スレッド名索引 | `~/.codex/session_index.jsonl` | 118 行 | 表示名からの候補拾い |
| ターン索引 | `~/.codex/thread_history_1.sqlite` | 225 thread / 1,796 turn | 索引と rollout の整合確認のみ |
| プロンプト履歴 | `~/.codex/history.jsonl` | 760 行 | 補助（有効ヒットなし） |

- 圧縮形式は `.jsonl.zst`（zstd）と無圧縮 `.jsonl` が混在。全件を展開して走査した。
- 補助索引（sqlite / session_index）は「rollout に無いスレッドが残っていないか」の
  整合確認にだけ使い、本文の一次情報は rollout とした。

## 2. 検索クエリと手法

一次走査（広め）:

```
27b-qsa | qsa | indexer | pentaCoxian | llama.cpp | Qwen3.8 | qwen3
```

二次走査（プロジェクト固有の別名・成果物名）:

```
qsa27b | Qwen3.8-27B-QSA | qsa_indexer | text-inference-speed | qsa-refusal
MM2 | orx/ | codex/qsa27b-mm2 | p0-dense-baseline | p1-indexer-distillation
```

三次走査（コミット・ブランチ）:

```
e697cf2c | cf0e697 | 6209237 | be2face | 733c672 | a968250 | 265d0a1
orx/p0-dense-baseline-oracle-capture | orx/p1-indexer-distillation-k-2048-r-4
orx/p2-16k-k-2048-retrieval-hf-persistence | codex/qsa-multimodal-export
codex/text-inference-speed | codex/mm2-multimodal-cache
```

手法上の注意:

1. 生の部分一致では誤ヒットが支配的（後述）。**100 文字以上の base64 風文字列を除去した
   うえで語境界で再走査**した。
2. ヒットした 30 ファイルを精読し、cwd・ユーザー発話・実行コマンドで関連度を判定した。
3. 判定は「リポジトリを実際に操作した」「QSA/indexer の設計・評価を議論した」のいずれかを
   満たすもの。cwd が一致するだけのセッションは除外した。

## 3. 誤ヒットと除外

| セッション | 見かけのヒット | 実際 |
|---|---|---|
| `01a0d137`（09/24） | qsa 2,772 | reasoning の `encrypted_content`（base64）内の偶然一致 |
| `01a1369a`（10/14） | qsa 16,813 | 同上（base64） |
| `01a0f7bb`（10/01） | pentaCoxian 19 | 音声ファイル `pentacoxian.wav` の TTS クローン。人物名の一致のみ |
| `01a182e0`（10/28） | pentaCoxian 2 | tailscale 端末一覧のユーザー名 |
| `01a06fb1`（09/05） | qsa 40 | 別モデル（K2-Horizon / Qwopus）の起動。QSA 議論なし |
| `01a19d7d`（11/03） | qsa 9 | ティザー画像制作。QSA 無関係 |
| `01a1a3b5`（11/04 08:18） | 27b-qsa 11 | cwd が 27B-QSA なだけ。内容は Modal の課金確認 |
| `01a1adbf`（11/06 07:06） | qsa 1 | Qwen の発表曜日グラフ。プロジェクト無関係 |

教訓: `QSA` は 3 文字で base64 に高頻度で現れる。**件数ではなく文脈で判定する**必要がある。

## 4. 本体セッション（時系列・JST）

| # | セッション ID | 期間（JST） | 役割 | 目的・決定・新事実 | 関連コミット |
|---|---|---|---|---|---|
| 1 | `01a19d82-5e4a-7412-862c-aa4774556103` | 11/03 03:25–03:56 | 計画の root | QSA（Qwen Sparse Attention）論文を調査し、Qwen3.8-27B への構造変換の可否、必要リソース（Modal/Colab）、実装後の性能概算、計画のクロスレビューまで実施。決定=最終的に Q8〜Q4 へ量子化し V100×1〜2 で稼働、目標は「長文でも速度が落ちない」快適性、着手は safetensors＋参照実装の小規模移植。cwd は `Documents/Codex/2026-11-03/a` | なし（会話のみ） |
| 2 | `01a19d94-3a5c-7f02-a711-212f2225e7e9` | 11/03 03:44–03:54 | fork: レビュー | 計画の独立クロスレビュー。「止めるべき P1 は無い」。P2 指摘=checkpoint に選択器重みだけでなく optimizer/scheduler/RNG/データ順序/step を保存、全トークン参照での一致は必要条件、未評価を成功とも失敗とも推測しない | なし |
| 3 | `01a19d94-71f7-7d92-89cb-26d3b90c4538` | 11/03 03:45–03:50 | fork: レビュー（別スレッド） | 同上の短縮版。長文評価が予算内に終わらなければ「未評価・証拠不足」と報告する規律を明記 | なし |
| 4 | `01a19d95-0975-71b2-bc8b-194a24eb6f75` | 11/03 03:45–03:51 | fork: 性能・コスト見積り | 追加 indexer **52,432,896 params（BF16 約 100 MiB）**。条件付き読出し削減比は 8k 1.005x / 32k 1.035x / 128k 1.151x / 256k 1.306x（Q4 重み+F16 KV では 128k 1.528x、256k 2.062x）。判定は「safetensors＋小規模移植・蒸留に条件付き GO」 | `work/research_notes/.../estimates.json` |
| 5 | `01a19d99-41aa-7960-a368-5bd2db2890b3` | 11/03 03:50–03:55 | fork: 実行計画書 | 計画を文書化。H200＋8 物理 CPU＋128 GiB で **$5.939856/h**、初回 profile 2 時間で約 **$11.88**、14 時間の段階 pilot で約 **$83.16**（gross 予算 $100）。128k の読出し比 1.151x（BF16）/1.528x（Q4_0）/1.283x（Q8_0） | `outputs/qsa_estimate.py` |
| 6 | `01a1a03e-23fc-7de2-83fc-74a1257fa5a0` | 11/03 16:09 → 11/06 03:35 | **本体（実装・評価・公開・配備）** | 下記 §5 に詳細。84 ユーザーターン、48 MB | §6 参照 |
| 7 | `01a1a3e5-7062-7bb0-93a8-3611f63c8b7b`（archived） | 11/04 09:11–09:15 | 運用確認 | 8092 をブラウザで開き約 30 万文字（約 856 KB）を投入、処理中であることを確認。長文投入時の実運用チェックのみ | なし |
| 8 | `01a1a769-721d-7753-bdce-42bc5496ec4a` | 11/05 01:34–01:42 | 図版（ティザー） | 27B-QSA のティザー画像をコード生成。**261,120 トークンで QSA 41.07 / Dense 22.91 tok/s（1.79 倍）**、16/64 層への適用、全 KV 保持、品質同等性は未評価である旨を図に明記。既存ティザースキルは使わない判断 | `docs/figures/qsa-teaser/` |
| 9 | `01a1a77d-2eb2-7f13-b8da-080652f87c83` | 11/05 01:56–01:57 | レビュー | 修正後アーキテクチャ図の独立レビュー。エンジン commit `cf0e697` の実装と照合（`server-context.cpp:605`、`qwen35.cpp:223`、`mtmd-helper.cpp:144`）。P1/P2 相当の誤りなし | 参照: `cf0e697` |
| 10 | `01a1a790-b590-7aa0-9114-b1097173fdb4` | 11/05 02:17 → 11/06 07:08 | 図解・スキル化・公開 | 図の検証→解説動画（字幕・音声・アニメーション）→ 既存フローチャートスキルをラップした **architecture-explainer** スキル化（small/mid/large の段階、配色、矢印と文字の衝突自動回避）→ 11/06 07:08 GitHub 公開 | 公開記録 `docs/architecture-explainer-publication.md` |
| 11 | `01a1a8b7-8cfb-7123-bcce-01d75261dce6` | 11/05 07:39–07:44 | 周辺 | Ripple-65B-Preview を同じ手法で QSA 化できるかの検討とアーキテクチャ図作成。**QSA 化自体は未実施** | なし |

fork（#2〜#5）はいずれも `parent_thread_id = 01a19d82`。同一ユーザーターンを
レビュー・見積り・文書化へ並列分割したもので、内容は重複する。

## 5. 本体セッション `01a1a03e` の詳細

期間 11/03 16:09 → 11/06 03:35（JST）。cwd `/home/ryu/OpenResearch/27b-qsa`。

### 5.1 品質・設計の判断

- 冒頭で P0/P1/P2 の結果をレビュー。**capture 0.9414 は最終学習系列の値で held-out ではない**
  と訂正（別入力の P0 は 0.9622）。PPL 7.07296 → 7.07176、retrieval 全問正解はログに実在。
- 「dense と区別できない」は言い過ぎで、正しくは「この規模では PPL 劣化を検出できなかった」。
- 方針決定: **画像・動画は元の dense attention、テキストのみ QSA**（16 層, K=2048, r=4）。
  **MTP 層は dense のまま**（学習済み indexer が無く、全キー読みが残る）。
- Stage 2（backbone 学習）より先に「速度を測れる QSA 推論実装」を優先する判断。

### 5.2 llama.cpp 移植と配備

- 当初はマスク参照版。262,144 コンテキストで起動成功、Q8_0 本体 29.3 GB＋indexer FP32、
  15,641 トークンで冒頭 passkey 正答。GPU 合計 約 47 GiB。
- 11/04「実験で MM2 を学習しました」を受け、**MM2 native cache port** を実装。
  block-table KV をカーネル内で解決し、選択ブロックだけを FP16 KV から読む。indexer の
  圧縮キーを永続化し、更新ブロックのみ再計算。prefix cache・MTP carry も checkpoint 化。
  decode 低下は最大 20% → 実用範囲へ圧縮。ベンチは `qsa91-live` のログを使用。
- 11/05 12:44、llama.cpp が既定で埋め込み層を CPU に置く挙動を VRAM へ変える PR 草案を作成
  （英語版・日本語版）。
- 11/05 14:30 V100 で 262k の Q4 版を起動。11/06 03:35 に Huihui Q4 QSA 版へ切り替え、
  8092（262,144 / MTP 最大 2 / BF16 mmproj）で稼働。

### 5.3 公開

- 11/04、HF（higa2515）と GitHub（kuraneko1）へ公開。エンジン差分と Q8_0 GGUF・indexer・
  元モデルを整備し、フォークの `codex/qsa27b-mm2` ブランチへ push。
- コミット `e697cf2c` "Preserve MM2 QSA runtime and source"。

### 5.4 量子化の評価

- unsloth の Qwen3.8-27B-GGUF に対し、**本体は再量子化せず indexer（FP32）だけをマージ**。
  UD-Q4_K_M 16.67 GB / UD-Q5_K_M 19.98 GB / UD-Q6_K 22.19 GB を作成し、全テンソル一致を検証。
  保存先 `/mnt/data2/qsa27b/gguf/unsloth-qsa-456bit-20261105/`。
- Q4_K_M は実文章 64k で **PPL +2.44%**、capture 73.82%、oracle との差 9.11 ポイント。
  **再学習が必要と結論づけるには別データでの再確認が必要**、というのが当時の判断。
- Indexer 自体は量子化していない（「Q4 化で変わる」のは本体が indexer へ渡す入力値）。

## 6. 前史・周辺セッション

| セッション ID | 期間（JST） | 位置づけ | 内容 |
|---|---|---|---|
| `01a06bb9-a858-7ff0-b421-736f693bc718` | 09/04 18:22 → 09/15 09:45 | 前史 | Qwen3.8-Flash-Next の MTP / FreeToken / QSA を調査し llama.cpp へ適用。**QSA はマイクロブロック単位で選択する**という理解、`qsa_sparse` バックエンド、paged KV・block-table の論点はここが土台。handoff 文書を作成 |
| `01a0bc04-146c-76f3-a4c6-690c23f9bc3a` | 09/20 08:32–08:55 | 前史 | `work/qwen38-llama-optimization` でボトルネック分析。**QSA indexer の文脈走査が深度項**（0.494 µs/文脈トークン、65k で +32 ms）であり、`LLAMA_QWEN4EXP_SPARSE_DECODE`（130k で +11%）は上流未移植のため移植候補と整理 |
| `01a18a19-1922-7fd0-a997-77a0c388d2c6` | 10/30 08:57 → 11/03 22:43 | 周辺 | `projects/z170/strata-distributed` のレビュー中に `native_qsa_indexer_append_steps`、`QsaIndexerBuffers` など QSA カーネル本体を読解。27B 実装の前提知識 |

## 7. 関連コミット・ブランチ

| コミット | 日時（JST） | 内容 |
|---|---|---|
| `a968250` | 11/03 07:24 | Initialize OpenResearch project |
| `4e44a1b` | 11/03 08:30 | QSA 参照実装・テスト・27B 適応計画 |
| `3816e91` | 11/03 08:38 | chunked teacher targets、論文図、実装状況 |
| `6352dd2` | 11/03 08:41 | 27B pilot の計算オプション実測 |
| `af73df5` | 11/03 08:49 | QSA attention patch と P0 dense capture baseline |
| `265d0a1` | 11/03 08:51 | 27B 重みを GPU へ直接ロード |
| `733c672` | 11/03 10:41 | `orx/p1-indexer-distillation-k-2048-r-4` |
| `be2face` | 11/03 13:37 | `orx/p2-16k-k-2048-retrieval-hf-persistence` |
| `e697cf2c` | 11/04 | `codex/qsa27b-mm2` "Preserve MM2 QSA runtime and source" |
| `cf0e697` | 11/05 以前 | アーキ図レビューが参照したエンジン実装の commit |
| `6209237df53569f07429f3f79fc5354de5fe704f` | — | 土台にした V100 fork の commit |

ブランチ: `codex/qsa-multimodal-export`、`codex/text-inference-speed`、
`codex/mm2-multimodal-cache`、`orx/p0〜p2`、`orx/mm0〜mm2`（`mm1-lr-1e-4`、
`mm2-16k-k-2048-300-step`、`mm2-qa-eval-v2〜v4` など）。

## 8. 網羅性の限界

1. **11/03 07:24–13:37 JST の初期コミット群を作った rollout が存在しない。**
   `a968250`〜`265d0a1` と `orx/p1`・`orx/p2` は、本ホストの Codex rollout の
   どのセッションにも実行痕跡が無い。`01a1a03e` は 16:09 開始で、冒頭に結果の要約が
   貼られているだけ。別ホスト（z170-desktop 等）または別エージェント実行の可能性が高い。
2. **11/06 の黒セット長文・拒否率 3arm の実験は Codex rollout に無い。**
   `CURRENT_STATUS.md` と `benchmarks/qsa-refusal-*` に記録があるが、実行主体は Codex 以外
   （DSH 側）と見られる。
3. **z170-desktop 側の `~/.codex` は未調査。** 本調査はこのホストの保存領域のみ。
4. 削除済み rollout は復元できない。sqlite 索引（225 thread）と rollout 248 件の突合では
   27B-QSA 固有の未回収スレッドは見つからなかったが、完全性は保証できない。
5. セッションの fork・アーカイブは重複して数えている（#2〜#5 は同一 root の fork、
   `01a1a3e5` は archived）。
6. 認証情報、トークン、無関係セッションの本文は意図的に収集・記載していない。

## 9. 再現手順

```sh
# 1) 全 rollout を展開して候補を粗く抽出
find ~/.codex/sessions ~/.codex/archived_sessions -type f \
  \( -name '*.jsonl' -o -name '*.jsonl.zst' \) | sort | while read -r f; do
  case "$f" in
    *.zst) zstd -dc -- "$f" ;;
    *)     cat -- "$f" ;;
  esac | grep -qiE '27b-qsa|qsa27b|qsa_indexer|pentacoxian|text-inference-speed' \
    && echo "$f"
done

# 2) base64 を除去して語境界で再走査（誤ヒット除去）
#    Python: re.sub(r'[A-Za-z0-9+/=_-]{100,}', ' ', text) の後に
#    r'\bQSA\b|27b-qsa|qsa27b|qsa_indexer' で検索

# 3) ヒットしたファイルのメタ情報と最初のユーザー発話を確認
#    session_meta（id / timestamp / cwd）→ response_item/message(role=user)
```

## 10. 付録: ファイルパス一覧

```
~/.codex/sessions/2026/11/03/rollout-2026-11-03T03-25-29-01a19d82-5e4a-7412-862c-aa4774556103.jsonl
~/.codex/sessions/2026/11/03/rollout-2026-11-03T03-44-59-01a19d94-3a5c-7f02-a711-212f2225e7e9.jsonl
~/.codex/sessions/2026/11/03/rollout-2026-11-03T03-45-13-01a19d94-71f7-7d92-89cb-26d3b90c4538.jsonl
~/.codex/sessions/2026/11/03/rollout-2026-11-03T03-45-52-01a19d95-0975-71b2-bc8b-194a24eb6f75.jsonl
~/.codex/sessions/2026/11/03/rollout-2026-11-03T03-50-29-01a19d99-41aa-7960-a368-5bd2db2890b3.jsonl
~/.codex/sessions/2026/11/03/rollout-2026-11-03T16-09-49-01a1a03e-23fc-7de2-83fc-74a1257fa5a0.jsonl
~/.codex/archived_sessions/rollout-2026-11-04T09-11-25-01a1a3e5-7062-7bb0-93a8-3611f63c8b7b.jsonl
~/.codex/sessions/2026/11/05/rollout-2026-11-05T01-34-28-01a1a769-721d-7753-bdce-42bc5496ec4a.jsonl
~/.codex/sessions/2026/11/05/rollout-2026-11-05T01-56-01-01a1a77d-2eb2-7f13-b8da-080652f87c83.jsonl
~/.codex/sessions/2026/11/05/rollout-2026-11-05T02-17-21-01a1a790-b590-7aa0-9114-b1097173fdb4.jsonl
~/.codex/sessions/2026/11/05/rollout-2026-11-05T07-39-24-01a1a8b7-8cfb-7123-bcce-01d75261dce6.jsonl
~/.codex/sessions/2026/09/04/rollout-2026-09-04T18-22-04-01a06bb9-a858-7ff0-b421-736f693bc718.jsonl.zst
~/.codex/sessions/2026/09/20/rollout-2026-09-20T08-32-59-01a0bc04-146c-76f3-a4c6-690c23f9bc3a.jsonl.zst
~/.codex/sessions/2026/10/30/rollout-2026-10-30T08-57-43-01a18a19-1922-7fd0-a997-77a0c388d2c6.jsonl
```
