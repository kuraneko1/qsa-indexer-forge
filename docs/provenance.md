# 来歴・固定 revision・SHA256

Qwen3.8-27B-QSA（llama.cpp native 実行系）の正本・成果物・学習履歴を 1 枚にまとめる。
**数値はすべてローカルで読める記録または計算による**。読めないものは「未確認」と明記する。

> 作成環境: ubuntu-desktop1729。`/mnt/data2` は未マウント、実 GGUF も実エンジンも無い。
> したがって SATA 上のファイルの存在・SHA256・サービス状態は**この文書では保証しない**。
> HF の公開状態・取得パスは委任元が HF API で確認したもの（2026-11-06）を含む。

---

## 1. 正本（canonical）の定義

| 対象 | 正本 | 補足 |
|---|---|---|
| 研究・実装・運用手順 | `/home/ryu/OpenResearch/27b-qsa` | git 管理。既定 `codex/mm2-multimodal-cache` |
| エンジン・ソース | 研究リポジトリの `release/engine-fork`（入れ子 git、外側から ignore） | 公開 GitHub と同一 commit を checkout。このリポジトリには含まない |
| エンジン差分 | `patches/*.patch` | base に単独適用。重ね禁止 |
| GGUF 変換 | `conversion/qwen.py`（engine 内）、`qsa/export_model.py` | モデル側の indexer 統合は前者 |
| 量子化済みモデル | HF（下表）と SATA `/mnt/data2/qsa27b/gguf/` | バイト列の正本は HF の固定 revision |
| 学習済み indexer | 公開 BF16 repo の `indexers/mm2_16k/qsa_indexers.pt`（§3.1） | 旧 repo は 401。SHA256 で照合する |
| 実行ログ・実験原本 | z170 の `/mnt/data2/qsa27b/` | 本機から未確認 |

境界: 外側 repo は「コード・手順・結果の要約」を追跡し、`release/engine-fork`、`release/payload`、
`release/original-metadata`、生 run、tar/zip、図の PNG/SVG は追跡しない（`.gitignore`、案 C）。
従って「エンジンの完全な履歴」は外側 repo ではなく engine-fork と patch が正本。

---

## 2. 関連ディレクトリ

### 2.1 本機（ubuntu-desktop1729）

| パス | 役割 | 状態 |
|---|---|---|
| `/home/ryu/OpenResearch/27b-qsa` | 正本リポジトリ | HEAD `99ba95c`、branch `codex/mm2-multimodal-cache` |
| `…/release/engine-fork` | 公開エンジンの checkout | 入れ子 git。HEAD `2cccda46`（PR #1 枝） |
| `…/release/payload` | 公開用ステージング（HF 用・runtime） | ignore。実体は HF と重複 |
| `…/release/engine-base.tar` | 公開時の base スナップショット | 177,274,880 bytes |
| `…/patches` | 3 段階 patch + graph 断片 | §4 |
| `…/qsa` | HF 参照実装（indexer / export / server） | 学習・変換の元 |
| `…/scripts` | 変換・移植・監査・起動 | `transplant-qsa-gguf.py` 等 |
| `…/configs` | 実行・学習の設定 JSON | `mm2-integration.json` 等 |
| `…/docs` | 実装・配備・制限の記録 | 本文書の主な出典 |
| `…/benchmarks`, `…/results` | 測定の設計・集計 | 生データは z170 |
| `…/models/Qwen3.8-27B-QSA` | 旧 text-only HF export | 18 shard + `qsa-indexers.safetensors` 209,750,584 bytes |
| `/home/ryu/OpenResearch/27b-qsa-quant` | 量子化ナレッジベースの器 | main の作業 tree は空。実体は OpenResearch worktree 側 |
| `/home/ryu/projects/QSA27B` | **IQ3_S 派生**の運用メモ（git 無し） | 本流とは別系統 |
| `/home/ryu/projects/qsa-refusal-20261106` | 拒否率実験の実行 repo | 5 commits、arm 起動のみ |
| `/home/ryu/projects/qsa-indexer-forge` | **この公開用リポジトリ** | Apache-2.0（`patches/` は MIT）。学習 tree `d75da83` と現行 export/runtime を統合 |

### 2.2 z170-desktop（SATA、本機から未確認）

| パス | 役割 |
|---|---|
| `/mnt/data2/qsa27b/llama.cpp` | v5 native（Q8、mask 参照版） |
| `/mnt/data2/qsa27b/llama.cpp-mm2` | MM2 版エンジン + `build-mm2-v1` |
| `/mnt/data2/qsa27b/llama.cpp-text-speed` | 速度比較用の独立ビルド |
| `/mnt/data2/qsa27b/engine-qsa` | **IQ3_S 派生**の公式エンジン |
| `/mnt/data2/qsa27b/ggml-runtime-v5` | 疎 FP16 カーネルを含む GGML |
| `/mnt/data2/qsa27b/gguf/` | Q8_0、mmproj、Unsloth 4/5/6bit、Huihui、IQ3_S |
| `/mnt/data2/qsa27b/models/` | BF16 + FP32 indexer 統合体 |
| `/mnt/data2/qsa27b/logs/`, `experiments/`, `releases/` | 実行ログ・実験原本・公開 snapshot |
| `/mnt/data2/qsa27b-recovery/2026-11-04-native-v5` | 旧 v5 のフル復元パッケージ |
| `/mnt/data2/qsa27b-recovery/development/mm2-cache-2026-11-04-r2` | MM2 差分 backup |
| `/home/ryu/Backups/27b-qsa/recovery-core-2026-11-04-r2.tar.gz` | 別機の小型復元 core（約 1.5GB） |

---

## 3. 公開リポジトリと固定 revision

| 種別 | リポジトリ | 固定 revision | 備考 |
|---|---|---|---|
| エンジン（本流） | [kuraneko1/Qwen3.8-27B-QSA-llama.cpp](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp) | `e697cf2c72ecaa165de3714ea6ac3b00649ab274` | 公開時の source。source 3,689 files 一致を検証 |
| エンジン base | [pentaCoxian/Qwen-V100-llama.cpp](https://github.com/pentaCoxian/Qwen-V100-llama.cpp) | `6209237df53569f07429f3f79fc5354de5fe704f` | patch の適用先。upstream の `qwen38-v100` tip も同 commit |
| エンジン（IQ3_S 派生） | [pentaCoxian/Qwen-QSA-llama.cpp](https://github.com/pentaCoxian/Qwen-QSA-llama.cpp) | `70b4e14a830a4ec78d1ac97087cd9fac02429d37` | **ローカルに object 無し（未確認）** |
| 研究リポジトリ | `github.com/kuraneko1/27b-qsa`（private） | HEAD `99ba95c` | 調査時は local 19 / origin 1（未 push 18）。旧記録は未 push 19 と記載 |
| BF16 + indexer | [higa2515/Qwen3.8-27B-QSA-BF16](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16) | `d61281f4626729e4cf138476a704c7fedc08f891` | 79 files、56,028,897,804 bytes |
| GGUF（Q8・runtime） | [higa2515/Qwen3.8-27B-QSA-GGUF](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF) | `b8bfeee64f93e4b7a619f798fc62df52847d5e42` | 15 files、Q8_0 + mmproj + runtime archive |
| 同上（Unsloth 追加後） | 同上 | `2bc63053a75be708507c0bacdc4088804ea54190` | Q4/Q5/Q6 の 3 files を追加した revision |
| indexer（step600、履歴） | `huggingface.co/higa2515/qwen38-27b-qsa-indexers` | `2ff995092226ca22b0d6c6ea2afc0bc85c0a5530` | 旧 repo。**現在 401（取得不可）**。revision は履歴識別子 |
| indexer（MM2 16k、履歴） | 同上 | `b51b8d15e4342f2bc11ad8419abc18e3be6a8c8a` | 同上。`mm2_16k/qsa_indexers.pt` という名前で記録 |
| **indexer（公開取得先）** | [higa2515/Qwen3.8-27B-QSA-BF16](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16) | `d61281f4626729e4cf138476a704c7fedc08f891` | **`indexers/mm2_16k/qsa_indexers.pt`**、SHA256 `e44193e5ef832b6220c8355fff92ac8f07d8c13538f584d2b6231b5dd2f4bc51`（§3.1） |
| 元モデル | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | backbone / vision / projector / MTP |
| Unsloth 量子化元 | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | UD-Q4_K_M / Q5_K_M / Q6_K |
| Huihui abliterated 元 | [huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | `1c69098e9c32db6dc06fc48f1d6cc75ad07da89e` | UD-DW-Q4_K_M |

改名履歴: `Qwen3.8-27B-QSA-MM2-{BF16,GGUF}` → `Qwen3.8-27B-QSA-{BF16,GGUF}`。
**revision・ファイル名・内容は不変**、旧 URL の転送を確認済み（`release/repository-renames.json`）。

### 3.1 indexer の取得経路（旧 repo は 401）

- 旧 `higa2515/qwen38-27b-qsa-indexers` は**現在 401**（非公開または権限なし）。
  したがって上の 2 つの revision は「いつ・どの重みを学習したか」を示す履歴識別子であり、
  **ダウンロード先として案内しない**。
- 公開されている最終 MM2 の checkpoint は
  `higa2515/Qwen3.8-27B-QSA-BF16 @ d61281f4626729e4cf138476a704c7fedc08f891` の
  **`indexers/mm2_16k/qsa_indexers.pt`**（HF API で確認、SHA256 `e44193e5ef832b6220c8355fff92ac8f07d8c13538f584d2b6231b5dd2f4bc51`）。
  README と `docs/training.md` の export 手順もこのパスを使う。
- 旧 indexer repo の `mm2_16k/qsa_indexers.pt` は同じ checkpoint の来歴上の名前。
  固定 revision `d61281f` では **`indexers/mm2_16k/qsa_indexers.pt`** が正しいパス。現在の main では `indexers/qsa_indexers.pt` へ移動しているが、再現手順では main ではなく固定 revision と、その revision に存在するパスを使う。
- step600（text-only）の `.pt` は旧 repo にしかなく、HF から再取得できるかは未確認。
  旧 text-only export の `qsa-indexers.safetensors` は研究リポジトリの
  `models/Qwen3.8-27B-QSA/` に残っている。

---

## 4. patch と SHA256

このリポジトリの `patches/`（MIT、`patches/LICENSE`）:

| patch | files | 差分 | SHA256 |
|---|---:|---|---|
| `llama-qsa27b.patch` | 14 | +196/−5 | `b59497f16406ef90d5cebcf73bd2e4b1cc1ad30e913b35113a46dd5d633c128f` |
| `llama-qsa27b-sparse.patch` | 17 | +488/−73 | `4531c8b561fa045041617a2fd0f8a59bab82c7b480ff465087c1b6f4660ab8a8` |
| `llama-qsa27b-mm2-cache.patch` | 22 | +605/−76 | `68f1c9e06077a7ac78b2fedf282aa899fff5f4f88afd7aa4768ee6f98d4e4ca5` |

研究リポジトリ側の参照物（このリポジトリには含まない）:

| ファイル | 内容 | SHA256 |
|---|---|---|
| `patches/qsa27b-graph.inc` | mask 経路の graph 断片（112 行） | `0438fab353f53be13f25afbb6db97486ab4a88aacd2ce77c334178e90f2da12f` |
| `release/github-review.patch` | 公開用の一式（README 差し替え含む） | `c9832b21077f93ff09b972264f0a65c4df08a77ac5a68ec624c93319090492fd` |

- 適用先はすべて `6209237`。**単独適用**、旧版と重ねない。
- 3 つの SHA256 は研究リポジトリの `patches/` と同一（本リポジトリへ複製後も一致を確認）。
- `llama-qsa27b-mm2-cache.patch` の SHA256 は `docs/mm2-native-cache-port.md:92-93`（研究リポジトリ）の記録と一致。
- 3 番目は公開 commit `e697cf2c` のコード部分（README / `qsa27b/` / validation を除く）に相当。

---

## 5. モデル成果物と SHA256

| 成果物 | サイズ | SHA256 | 出所 |
|---|---:|---|---|
| Q8_0 本体（MM2） | — | `97c4ecdd175664b056df1822a4e2a93e504a5a7fc8789a3a5cd51517f73c711f` | `docs/mm2-native-cache-port.md` |
| mmproj F16 | 927,606,912 | `e675225a5f734f568ecdf7bdfa9ed03358be6fa2a5783a8c17f9295c82f0e028` | `docs/z170-deployment.md` |
| runtime archive（sm70） | 96,816,785 | `4ea89a6260be0f289993896c396f9f20d5d1493e24f35b592362634f19a89748` | `release/publication.json` |
| Unsloth UD-Q4_K_M + QSA | 16,674,185,792 | `30f12055d5f3272381a2e6ee8ad2281cafa907373f3b34a80036cc2a992d4707` | `release/unsloth-456/publication.json` |
| Unsloth UD-Q5_K_M + QSA | 19,981,255,232 | `63727171d5c5d2ec5cae28f9573a512bbc707ada1aaad6ac8e275123c2fefebf` | 同上 |
| Unsloth UD-Q6_K + QSA | 22,193,422,912 | `8e47564fe7d8eb9e6e145324f19752314720425ea5770304a7234bd7f7afbd51` | 同上 |
| Huihui UD-DW-Q4_K_M + QSA | 16,761,061,952 | `5437210a2313e3777f17d97c06349932ea63cb13853af9b762c892114d0c0ab5` | `results/huihui-…/SHA256SUMS` |
| Huihui 元 mmproj | — | `c9a09064683620bea3d3bfed5d4462e1a97a7d2fff7e5045d6862a0a85eeb5b5` | 同上 |
| MM2 indexer checkpoint | 209,765,130 | `e44193e5ef832b6220c8355fff92ac8f07d8c13538f584d2b6231b5dd2f4bc51` | `configs/mm2-integration.json` |
| 旧 text-only indexer safetensors | 209,750,584 | （ローカル計算なし） | `models/Qwen3.8-27B-QSA/` |

注意:

- サイズ・SHA は「その時点のファイル」の値。HF の revision を固定して取得し、
  `sha256sum -c` で照合してから使う。
- Q8_0 の **29,256,829,760 bytes** は旧 v5 の `Qwen3.8-27B-QSA-Q8_0.gguf` の値。
  MM2 版（`Qwen3.8-27B-QSA-mm2-Q8_0.gguf`、SHA `97c4ecdd…`）は別ファイルなので、
  このサイズと結び付けない。
- indexer の公開取得は §3.1 の `indexers/mm2_16k/qsa_indexers.pt` を使う。旧 repo は 401。

---

## 6. 学習 run と indexer

| 項目 | 値 |
|---|---|
| 最終 text indexer 学習 run | `57dd1e1d-54c5-4a62-95ab-b9fb47341056` |
| 学習 commit | `be2face7d7e40241c8efbb9be70f5d6e54df64d1` |
| checkpoint（履歴） | `warm100/qsa_indexers_step600.pt`（旧 repo。現在は取得不可） |
| checkpoint SHA256 | `9a97ef6d385130c59734515d88a968cb423dcd6153653348f69386cc58a2d26c` |
| MM2 継続蒸留 run | `9dd0ce84-f7f2-46a6-92bb-33a915fc7ae8` |
| MM2 学習 commit | `d75da83` |
| MM2 設定 | 16,384 tokens × 300 steps、lr 3e-4、text/image 0.5、backbone/vision/projector/MTP 凍結 |
| MM2 checkpoint（公開） | `indexers/mm2_16k/qsa_indexers.pt`（BF16 repo `d61281f`） |
| MM2 checkpoint（履歴名） | `mm2_16k/qsa_indexers.pt`（同じ checkpoint） |
| DocVQA probe run | `99942b8e-280f-4905-96aa-f1494530ce45` |
| answer NLL run | `87d47dd8-0b52-4fbc-84b1-7e15477519b2` |

学習の再現性に関する制約（`docs/recovery.md`）: 当時の学習データは streaming 取得で revision と
全入力トークンを保存しておらず、optimizer / RNG 状態も checkpoint に含まれない。
**同一ビット列の重みを学習から再現することは保証できない**。保存済み重みを使う。

---

## 7. IQ3_S 24GB 派生（本流と区別）

RTX 3090 24GB 単体で動かすための**別系統**。本流（Q8_0 / UD-Q4〜Q6、MM2、MTP）とは
エンジンもモデルも配布元も異なる。混同しない。

| 項目 | 本流 | IQ3_S 派生 |
|---|---|---|
| モデル | `Qwen3.8-27B-QSA`（Q8_0 / UD-Q4_K_M / Q5_K_M / Q6_K、MM2 indexer） | `pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF` の IQ3_S（12.3GB + 1.0GB + 0.9GB） |
| エンジン | この fork `e697cf2c`（base `6209237`） | `pentaCoxian/Qwen-QSA-llama.cpp @70b4e14` |
| 実体 | 研究リポジトリ `release/engine-fork`、z170 `llama.cpp-mm2` | z170 `/mnt/data2/qsa27b/engine-qsa` |
| 投機 | 内蔵 MTP（draft 2） | DFlash2（別 GGUF）+ MTP 切替可 |
| 起動 | `qsa27b/launch-captured-runtime.sh` | `engine-qsa/deploy/qsa27b-24gb/serve.sh`（unit から env 制御） |
| KV | f16（実験では Q8_0） | `-ctk q8_0 -ctv q8_0`、draft KV f16 |
| 対象 HW | sm70 2×32GB（公開 archive） | sm70 + sm86 ビルド、3090 単体 24GB |
| 記録 | `docs/`、`patches/` | `/home/ryu/projects/QSA27B/DEPLOY-IQ3S.md` |
| SHA256 | §5 に記載 | **ローカル記録に無い**（doc は「照合済み」とのみ） |
| 状態 | 公開・検証済み | 運用メモのみ。エンジン commit は未取得 |

IQ3_S の 24GB 向けチューニング（層分割なし、VRAM 余裕 1GiB 弱、dflash 4 draft が最良など）は
派生側の記録であり、本流の設定に混ぜない。

---

## 8. 未確認事項（明示）

1. `/mnt/data2/qsa27b/` 配下の実ファイル・SHA256・サービス状態（本機に mount が無い）。
2. `pentaCoxian/Qwen-QSA-llama.cpp @70b4e14` の実体（`git cat-file` 不可、fetch 失敗）。
3. IQ3_S 3 ファイルの SHA256（ローカル記録に値が無い）。
4. 3 patch は clean `6209237` archive への `git apply --check` 済み。実適用後の CUDA ビルド・推論は未実行。
5. 公開 archive の再ビルドが同一バイナリになること（記録でも保証していない）。
6. CPU / Vulkan など CUDA 以外での QSA 性能・精度。
7. 262,144 の実入力、128k/262k のマルチモーダル品質、BF16 同条件対照。
8. step600（text-only）の `.pt` を現在の HF から取得できるか（旧 repo は 401 のため）。
9. 数値の評価条件と留保は `docs/evaluation.md` を正とし、本書は revision と SHA の索引に限定する。

---

## 9. 更新ルール

- revision は短縮せず 40 桁で書く。patch は SHA256 を併記する。
- 「実行した」と「記録にある」を分ける。実行していないコマンドを実行済みと書かない。
- 数値には条件（context、KV 型、量子化、MTP、GPU）を付ける。条件の違う表を混ぜない。
- 本流と IQ3_S 派生を同じ表・同じ結論にまとめない。
