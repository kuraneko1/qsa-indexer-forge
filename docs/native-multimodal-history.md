# MM2 QSA: 速度設定の継承と画像・動画キャッシュの移植

> これは原研究の履歴写しです。現在の公開手順は [llama.cpp 対応ガイド](llama-cpp.md)、評価条件は [評価と限界](evaluation.md) を参照してください。文中のサーバー・復元コマンドは当時の記録であり、そのまま実行しないでください。

2026-11-04。開発ブランチ `codex/mm2-multimodal-cache`。
テキスト速度比較で採用した安全な最速設定を土台に、学習済みMM2を画像・動画にも適用した。
作業開始時の8091の直接起動プロセスを終了し、新しい `qsa27b-mm2.service` に置き換えた。
旧 `qsa27b-llama.service` と旧ビルド・重みは保持、停止中。自動起動は新規に有効化していない。

## 稼働構成

| 項目 | 値 |
|---|---|
| 画面 / API | http://100.81.30.125:8091/ / http://100.81.30.125:8091/v1 |
| 待受 | `0.0.0.0:8091`、既存tailnet・SSH転送を維持 |
| モデル名 | `qwen3.8-27b-qsa`、旧名 `qwen3.8-27b-qsa-text` と `qwen3.8-27b-qsa-mm2` も受理 |
| 本体 | `/mnt/data2/qsa27b/gguf/Qwen3.8-27B-QSA-mm2-Q8_0.gguf` |
| mmproj | `/mnt/data2/qsa27b/gguf/mmproj-Qwen3.8-27B-QSA-F16.gguf`、変更なし |
| エンジン | `/mnt/data2/qsa27b/llama.cpp-mm2/build-mm2-v1/bin/llama-server` |
| GGML | `/mnt/data2/qsa27b/ggml-runtime-v5/lib`、既存の疎CUDAカーネルを継承 |
| 起動 | `/mnt/data2/qsa27b/src/scripts/serve-llama-mm2-z170.sh` |
| unit | `~/.config/systemd/user/qsa27b-mm2.service` |
| ログ | `/mnt/data2/qsa27b/logs/llama-server-mm2.log` |
| 配置 | SATA SSD `/mnt/data2/qsa27b` |
| GPU | CUDA0 V100-PCIE-32GB / CUDA1 PG500-216 32GB、layer split 1:1 |
| 実行 | 262144 context、1 slot、batch2048 / ubatch512、FP16 KV、MTP最大2トークン |
| QSA | 16全注意層、MM2 FP32 indexer、4トークン単位、K=2048 + 自ブロック |

主な言語モデル行列は従来どおりQ8_0。新しい量子化方式・再学習は行っていない。
Kは固定なので、すべての長さで12.5%を見る設定ではない。
HF/BF16参照版の `multimodal_policy=dense` は変更していない。
今回の画像・動画QSAは専用native実装の動作であり、通常のllama.cppでは再現できない。

## 実装したこと

1. テキスト・画像・動画を同じQSA経路に統一した。画像を追加しただけで
   本体・indexer・MTPのprefix cacheを全消去する切替をなくした。
2. indexerのプーリングキーの位置に、KVセルが保持する実際のM-RoPEのt/y/xを使う。
   画像パッチの物理的な行番号をそのまま位置とする処理を参照経路から除いた。
3. 元のMTP実装が画像embedding入力をスキップしていた点を修正した。
   視覚embedding Eと、一つずらした本体hidden Hを別々に渡し、画像の位置情報を保って
   MTP側のKVも更新する。microbatch分割後も元のEの行を選べるようにした。
4. MTPの `pending_h` を既存checkpointのspeculative状態に保存・復元する。
   会話を新規開始するときはcarryをリセットする。
5. Qwen35にもMTPのKV-only更新を許可した。以前はフラグを指定しても内部ガードが
   Qwen4Expだけを許可していたため、27Bでは不要なquery計算が残っていた。
   以前の検証は更新結果の一致までであり、最適化が実行された証拠にはなっていなかった。
   今回はquery projectionが評価されないこともfixtureで確認した。

MTP lag / asyncは0。以前の速度開発でcheckpoint復元後の欠落を起こした設定は採用していない。
MTPそのものはdense attentionを保つ。backbone・vision encoder・mmproj・MTP重みは凍結済みのものを使用する。

## 検証結果

記録は `results/mm2-native/`、元ログはz170の `/mnt/data2/qsa27b/logs/mm2-*`。

* GGUFの80個のFP32 indexerテンソルはMM2 HF版と完全一致。
  残り866テンソル（29,036,089,344 payload bytes）は旧Q8 GGUFとバイト単位で同一。
* 小型M-RoPE fixture：dense、疎カーネル、マスク参照を各microbatch7/64で検証。
  HFとのargmaxは全6条件29/29一致、最大logit差は0.000653以下。
  checkpointを復元し、未来のKVを除去して再実行した結果も一致した。
* MTP fixture：KV-onlyの次token logits差0、query projection未評価。
  画像embeddingの別入力とmicrobatch3の分割でも、通常token embedding経路との次logits差0。
* 27B dense / sparseの両方で、8k画像色、16k OCRの6桁コード、8k表、8k動画色、
  テキスト応答が正答。これは限定的な動作確認であり、包括的な画像・動画評価ではない。
* 画像付き追質問：8,470トークン再利用、再計算26トークン、全応答0.508秒。
  新画像追加では8,499トークン再利用し、新しい赤画像を正答。
  画像差替えでは共通テキスト7,951トークンを再利用し、差替え後の赤を正答。
  画像の後で新規テキスト会話を開始するケースも正答。
* tailnetの公開URLに対し旧モデル名でSSEを実行し、逐次出力・usage・`[DONE]` を確認。
  64kの画像付き会話と、その画像に対する追質問も通った。

| 測定 | 生成速度 | 入力処理 | 備考 |
|---|---:|---:|---|
| 16k画像・dense | 39.95 token/s | 1128 token/s | 同一実装のdense対照、128出力 |
| 16k画像・MM2 QSA | 42.88 token/s | 1155 token/s | 128出力、cache無効 |
| 4k画像・MM2 QSA / SSE | 41.99 token/s | 897 token/s | 256出力、4,392入力、cacheなし |
| 64k画像・MM2 QSA / SSE | 40.69 token/s | 1165 token/s | 256出力、65,827入力中3,876を再利用 |
| 64k画像への追質問 | — | 28トークンのみ処理 | 66,082再利用 / 66,110入力、初回本文1.18秒、全応答1.24秒 |

各条件1回の合成入力で、4k→64kの生成速度低下は約3.1%。反復測定・信頼区間はない。
生成内容とMTP採択率も異なるため、改善率を一般化できない。
64kでの初回本文までは約53秒かかった。キャッシュ再利用は初回の長文読込を省けるものではない。
262kは設定・割当を確認しているが、今回のMM2の画像付き品質検証は64kまで。
64kの画像確認は単純な色の質問で、長文OCR・複数文書・複雑な動画の精度は未確定。
QSAは全KVを保持し、KV容量削減の方式ではない。

## 来歴とハッシュ

* Backbone: `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
* Indexer: `higa2515/qwen38-27b-qsa-indexers` @ `b51b8d15e4342f2bc11ad8419abc18e3be6a8c8a`、
  `mm2_16k/qsa_indexers.pt`、SHA256 `e44193e5ef832b6220c8355fff92ac8f07d8c13538f584d2b6231b5dd2f4bc51`。
* エンジン土台: `pentaCoxian/Qwen-V100-llama.cpp` @ `6209237df53569f07429f3f79fc5354de5fe704f`。
* 全差分: `patches/llama-qsa27b-mm2-cache.patch`、SHA256
  `68f1c9e06077a7ac78b2fedf282aa899fff5f4f88afd7aa4768ee6f98d4e4ca5`。
  旧v5パッチへの追加差分ではなく、上記baseに一回適用する全差分。
* MM2 Q8 GGUF: SHA256 `97c4ecdd175664b056df1822a4e2a93e504a5a7fc8789a3a5cd51517f73c711f`。
* mmproj: SHA256 `e675225a5f734f568ecdf7bdfa9ed03358be6fa2a5783a8c17f9295c82f0e028`。

MM2の学習・DocVQA評価の詳細と限界は `configs/mm2-integration.json`。
今回新しくindexerを学習したものではなく、orxで学習されたMM2を、これまでのQSA移植と
FlashNext用V100 forkの速度開発の延長として組み込んだ。

## 運用とキャッシュの範囲

```sh
ssh z170-desktop1729 'systemctl --user status qsa27b-mm2 --no-pager'
ssh z170-desktop1729 'systemctl --user restart qsa27b-mm2'
ssh z170-desktop1729 'curl -fsS http://127.0.0.1:8091/health'
```

既存1スロット内の連続会話・共通prefixを再利用する。画像は媒体hashも照合するため、
別画像に古い画像KVをそのまま適用しない。RAM上のKV・indexer pooled key・MTP carryと
同一プロセス内checkpointが対象で、サーバー再起動後のディスク永続cacheではない。
`--cache-ram 0`、`--ctx-checkpoints 8`、`--checkpoint-min-step 1024`、context shift無効。
複数会話を同時に常駐させる構成ではなく、他会話がスロットを置き換えれば再利用範囲は減る。

## 空ディレクトリからの復元

旧v5を凍結した完全バックアップと、今回のMM2差分バックアップの2点が必要。
同一SSD内バックアップはSSD故障には耐えない。MM2の小型coreは別マシンにも保存するが、
29GBのMM2 GGUFと旧完全バックアップ全体の別装置保管は別途必要。

* baseline: `/mnt/data2/qsa27b-recovery/2026-11-04-native-v5`。
  inventory SHA256 `a8264116d82c04328da52b6b3af6b5c751a131e16e309ed4b4cf1e882ca7e9c9`。
* MM2 overlay: `/mnt/data2/qsa27b-recovery/development/mm2-cache-2026-11-04-r2`。
  `MANIFEST.json` が全ファイルのハッシュとsymlink先を保持。
  `scripts/capture-mm2-recovery.py` が独立コピーで作成する。
* overlayはMM2 GGUF、元のMM2 checkpoint、MM2 HF追加重み・設定、実行バイナリと共有ライブラリ、
  native base git bundle、全パッチ、起動設定、プロジェクトソース、検証結果・ログを保存する。
  変更していない18個のBF16 backbone shard・mmproj・GGML・CUDA・venvはbaselineから復元する。

まず `docs/recovery.md` の手順でbaselineを空のrootへ復元する。
本番rootを上書きせず、最初は別ディレクトリで行う。以下の `restored_root` はその復元先。

```sh
mm2_backup=/mnt/data2/qsa27b-recovery/development/mm2-cache-2026-11-04-r2
restored_root=/path/to/new/qsa27b
cp "$mm2_backup/gguf/Qwen3.8-27B-QSA-mm2-Q8_0.gguf" "$restored_root/gguf/"
cp -a "$restored_root/models/Qwen3.8-27B-QSA" "$restored_root/models/Qwen3.8-27B-QSA-mm2"
cp -a "$mm2_backup/models/Qwen3.8-27B-QSA-mm2/." "$restored_root/models/Qwen3.8-27B-QSA-mm2/"
cp -a "$mm2_backup/indexers" "$restored_root/"
cp -a "$mm2_backup/project/." "$restored_root/src/"
git init "$restored_root/llama.cpp-mm2"
git -C "$restored_root/llama.cpp-mm2" bundle unbundle "$mm2_backup/native-base.bundle"
cp "$mm2_backup/records/native-shallow.txt" "$restored_root/llama.cpp-mm2/.git/shallow"
git -C "$restored_root/llama.cpp-mm2" checkout 6209237df53569f07429f3f79fc5354de5fe704f
git -C "$restored_root/llama.cpp-mm2" apply "$mm2_backup/project/patches/llama-qsa27b-mm2-cache.patch"
mkdir -p "$restored_root/llama.cpp-mm2/build-mm2-v1"
cp -a "$mm2_backup/runtime/bin" "$restored_root/llama.cpp-mm2/build-mm2-v1/"
QSA_ROOT="$restored_root" QSA_HOST=127.0.0.1 QSA_PORT=8092 \
  bash "$restored_root/src/scripts/serve-llama-mm2-z170.sh"
```

元のnative repoはshallow取得のため、通常の`git clone bundle`では親commit不足になる。
上の手順は境界情報も復元する。r1のcaptureはこの手順の検証で不備を検出したため採用しない。
これは互換Linux/NVIDIA/CUDA環境での復元。rootを変更した場合はsystemd unitの
WorkingDirectory/ExecStartとQSA_ROOTも変更する。元rootなら保存済みunitを配置し、
`systemctl --user daemon-reload` 後に開始できる。旧サーバーとのGPU同時使用を避ける。
再ビルドは保存したbase+全パッチに対し、baselineからのGGML v5を使う。

```sh
cmake -S "$restored_root/llama.cpp-mm2" -B "$restored_root/llama.cpp-mm2/build-mm2-v1" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_USE_SYSTEM_GGML=ON \
  -DCMAKE_PREFIX_PATH="$restored_root/ggml-runtime-v5" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=ON
cmake --build "$restored_root/llama.cpp-mm2/build-mm2-v1" --target llama-server -j8
```

検証スクリプト: `validate-mm2-artifacts.py`（重み）、`check-mm2-mtp.cpp`（MTP）、
`check-mm2-mrope.cpp` + `check-mm2-mrope.py`（位置と復元）、
`validate-mm2-runtime.py`（別ポートのdense/sparse比較）、`probe-mm2-production.py`（公開SSEと64k）。
27B比較は両GPUを使うため、productionと同時に起動しない。

復元確認済みcore: `mm2-cache-2026-11-04-r2-core.tar.gz`（約423MiB）。
SHA256 `4ec85c1470c2ad41943e79140fab7a8458cda3fc81e349a26abb239aab1f03d4`。
196項目の照合、shallow境界を含むnativeソース復元、全パッチ適用、
展開先の共有ライブラリによるバイナリ起動が通った。
検証記録は `results/mm2-native/mm2-recovery-validation.json`。
手元の `/home/ryu/Backups/27b-qsa/` にcoreの同一コピーを保持する。
coreには29GBのGGUFを含めず、SATA側の完全overlayとv5 baselineが必要。
