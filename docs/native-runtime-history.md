# FlashNextの実行経路を27Bへ移す

> これは原研究の履歴写しです。現在の公開手順は [llama.cpp 対応ガイド](llama-cpp.md)、評価条件は [評価と限界](evaluation.md) を参照してください。文中のサーバー・復元コマンドは当時の記録であり、そのまま実行しないでください。

2026-11-03。対象は同じQ8 GGUF、V100系32GB 2枚。モデルの重みは変更しない。
テキストはQSA、画像・動画を含む会話はdense。tailnetの8091番で提供する。

## 移植の範囲

FlashNext技術報告の中心は、GDNとQSAの併用、圧縮されたキーに対する選択、
選択ブロックを直接読むカーネル、MTPの利用である。報告の高速化率はattentionモジュールの
測定であり、27B全体の速度を保証しない。indexerの文脈長依存の走査も残る。

出典: [On the Design of Qwen3.8-Next Architecture, sections 2.1.2 and Efficiency Analysis](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2)

27Bはdense FFNのモデルで、FlashNextのMoE・Gated Residual・n-gram重みを持たない。
重みとマルチモーダル機能を保持するため、それらの演算へ置き換えてはいない。
移植対象はQSAとMTPの実行経路。

## 実装

- FlashNext用に既存forkにあったブロック疎処理を`llm_build_qsa_base`へ共通化。
  Qwen35とQwen4expの両方が同じ処理を使用する。
- indexerキーは学習時と同じ順序で、射影、トークンごとのRMSNorm、r=4平均、
  ブロックRMSNorm、RoPEを適用。圧縮済みキーをキャッシュし、更新したブロックだけを再計算する。
- 既存Q8 KV用CUDA疎カーネルをFP16 KVにも対応させた。選択ブロックのK/Vを直接読む。
  長文で全KVを読み込むマスク方式や、クエリ別の巨大なK/Vコピーを避ける。
- 完全ブロック境界でも自分のブロックを含める、今回のpilotの動作を保持。
  自分のブロックがtop-kにもある場合は重複をマスクする。
- 短文、未対応デバイス・形状、無効化されたキャッシュでは参照経路を使う。
- 短いprefill（既定8k以下、16クエリ超）は同じ選択をFA用マスクへ展開する。
  decodeはブロックを直接読む。prefillの選択を省略する変更ではない。
- 同じ処理経路の会話はprefix cacheを再利用する。dense/QSAを切り替える際は
  target・draft両方のキャッシュを破棄する。1スロット、context shift無効。
- 元の1ヘッドのMTP重みをロードし、1トークンのdraftと本体による検証を有効化。
  1/2/3トークンを短文で比較し、長文での追加走査を抑えるため既定を1にした。
- Qwen35 MTPのmasked hidden-state出力を修正。複数行のうち必要な行が先頭へ集約される前に
  contextが先頭行を読み取っていたため、指定された行をFFNの前で選ぶ。
- MTPのprompt catch-upはK/V保存だけを行えるようにした。既存ドライバーのKV-only設定を利用する。
- MTPの4行paddingを有効にし、draftのグラフ再構築を減らす。
- 64k以上のMTPでは、paddingした全行のK/Vを書きつつ、必要な1行のQ・gate・注意出力だけを計算。
  元のキャッシュの可視性規則で1列のマスクを作る。疎カーネルへ**全ブロック**のIDを渡し、
  全キーへの注意を計算する。大きなリストはCUDAの分割数を増やす。
  穴のあるキャッシュ、位置の飛び、M-RoPEの可視性は既存のマスク生成に従う。

MTPの正規化後のhidden stateを次段へ渡す契約は、
[vLLMのQwen3.5 MTP実装](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)
とも照合した。

27BのMTP層には学習済みindexerがない。MTP層自身はdense attentionを維持している。
FlashNextの「MTP層のQSAとdraft段間でのindex再利用」はこの重みでは未実装。
既定ではMTPの注意範囲を近似しない。重みを変えずに実装できる計算上の無駄を削った段階。
主モデルは各クエリの学習済みindexerで選択する。

## 数値検証

- F16疎カーネル: D=256、24Q/4KV、r=4、513ブロック、1/4/32クエリ、
  1/17分割。重複・部分ブロックを含め、CPU基準との最大誤差は約1.1e-7。
- 小型モデル: 学習済みと同じindexer形状・本体attention形状で29トークンを分割処理。
  参照と高速経路の選択ブロックIDが一致。logit最大差約5e-4、argmaxは29/29一致。
  融合カーネルと既存FAのF16丸め方が異なるため、bitwise一致は要求していない。
- 元のF32小型モデルのHF比較: logit最大差約2.1e-7を維持。
- MTPの行選択: 修正前の小型テストの最大差3.6775、修正後は0。
- MTP KV-only catch-up後の次トークンlogit: 通常経路との差0。
- MTPの必要な1行だけを計算する経路: 不連続な位置と空きセル、未来のpaddingを含む
  小型モデルで通常FAと比較。hidden最大差0.000367、logit最大差0.000115、argmax一致。
  `LLAMA_MTP_DENSE_BLOCK_MIN=0`でこの経路を短文でも強制して検証。
- 65トークン・ubatch=32のprefill切替: 参照に対する最大logit差2.9e-6。

テストは研究プロジェクトの`scripts/`に置き、llama.cppの`tests/`には追加していない。
実機ログは`/mnt/data2/qsa27b/logs/`に保存。

## 測定方法と限界

`scripts/bench-native-qsa.py`で同じテンプレートと文章を使い、入力トークン数を固定。
temperature=0、seed=42、128出力トークンで測定する。
独立したdense/QSA比較は`cache_prompt=false`。長さ別の最終曲線は`--reuse-prefix`で
共通prefixを再利用し、追加で処理したトークン数・時間とcache数を明示する。
生成速度と初回の入力処理を分け、MTPの提案数・受理数も記録する。
文脈によって生成文とMTP受理率が変わるので、単発の速度比だけで一定速度を保証しない。

暫定の目標は、同じ27B・Q8で4kから長文へ伸ばした場合の生成速度低下10%以内。
初回に読むトークン数自体が増えるため、入力を読む総時間は別の指標とする。

Q8後の本格的なPPL、広範なマルチモーダル評価、複数seedの品質比較は未実施。
長文passkeyの正答だけで、一般的な長文品質の同等性を主張しない。

## QSA単独の比較

MTPを切り、同じQ8 GGUF・FP16 KV・GPUで比較したv3の結果。
各条件1回、128トークン生成。短文のQSA追加コストと長文での利点を分けて見る。

| 実行経路 | 4,096生成 tok/s | 65,536生成 tok/s | 長文による低下 |
|---|---:|---:|---:|
| dense、MTPなし | 25.35 | 19.46 | 23.2% |
| QSA、MTPなし | 24.28 | 23.88 | 1.6% |

64kではQSAがdenseより22.7%速い。4kでは4.2%遅い。
この比較ではQSAの生成速度低下を抑える効果を確認できた。

ただし**初回の入力処理はまだ遅い**。同じMTPなし比較で、4kはdense 1,124/QSA 852 tok/s、
64kはdense 888/QSA 797 tok/s。indexer・top-k・疎カーネルのprompt処理が負担になる。
生成が安定したことを、初回入力も一定時間で読めるという意味には使わない。

## MTPを含む現行v5の結果

同じQ8 GGUF、FP16 KV、V100系2枚、MTP提案長1。128トークンを生成。
長さを順に増やした1回の曲線。入力時間は**その行で新たに処理した分**であり、
262kをゼロから読む時間ではない。

| 総入力トークン | prefix cache | 新規処理トークン | 新規入力の処理時間 | 生成 tok/s |
|---:|---:|---:|---:|---:|
| 4,096 | 0 | 4,096 | 5.12秒 | 34.85 |
| 16,384 | 3,580 | 12,804 | 16.96秒 | 36.68 |
| 65,536 | 15,868 | 49,668 | 69.76秒 | 35.25 |
| 131,072 | 65,020 | 66,052 | 105.10秒 | 33.82 |
| 262,000 | 130,556 | 131,444 | 249.46秒 | 32.27 |

この曲線だけなら4k→262kの低下は7.4%だが、短文側も速度が変動した。
同じ入力・同じseedで各3回、追加測定した結果を主な判定に使う。

| 総入力 | 生成速度の範囲 tok/s | 中央値 tok/s |
|---:|---:|---:|
| 4,096 | 36.03–37.43 | 36.97 |
| 262,000 | 31.86–32.71 | 31.87 |

**中央値で13.8%低下。「速度低下ほぼなし」は未達。**
3回は信頼区間を出すための十分な標本ではなく、複数seed・複数文書の測定も必要。
MTP受理率は長文3回で54/72、52/74、52/75、短文で55/72、52/74、54/73。
生成内容・丸め・受理率の変化を含む実測であり、全差分をカーネルだけに帰属させない。

繰り返し測定ではprefixを再利用し、262kで各261,996トークンがcache済み。
残り4トークンでも入力処理に約1.24秒かかった。履歴復旧時の圧縮キー再構築などが残る。
初回入力・会話の初動遅延・生成速度を別々に改善する必要がある。

最終v5で262,000トークンの冒頭passkey `731924`を正答。
文章2,418トークンで`OK`、青い画像`Blue`、赤い動画`Red`、媒体後のテキスト、SSEも通過。
実機ログ: `bench-sparse-mtp-v5.json`、`bench-sparse-mtp-v5-repeat262.json`、
`bench-sparse-mtp-v5-repeat4k.json`、`passkey-sparse-v5-262000.json`、`native-v5-smoke.json`。
要約データ: `results/native-runtime-2026-11-03.json`。

次に進む判断は、**この版を実験用の稼働版として保持し、速度目標達成版とは扱わない**。
次の対象はMTPでの選択ブロック再利用とindexer・prefill処理の計測／最適化。
MTPの注意範囲を変更する実験では、targetによる検証と受理率・速度を比較する。
本体の学習済みindexerの選択を変える実験とは区別して品質を評価する。

## 再現・復旧

- 元のfork: `6209237df53569f07429f3f79fc5354de5fe704f`
- 現在の変更: `patches/llama-qsa27b-sparse.patch`。cleanな同commitへ単独で適用する。
  以前のマスク版パッチと重ねて適用しない。
- 起動: `scripts/serve-llama-sparse-z170.sh`
- CPU/C++ビルド: `/mnt/data2/qsa27b/llama.cpp/build-v5`
- CUDAライブラリ: `/mnt/data2/qsa27b/ggml-runtime-v5`
- CUDAの変更は`qsa-attn.cu`と`qsa-attn-kernels.cuh`のみ。
  `scripts/build-qsa-cuda-incremental.py --version v5`は同commitの既存CUDAオブジェクトを読み取り専用で再利用し、
  変更したオブジェクトを独立してビルド・リンクする。
- 元のビルドがない場合は、パッチ済みforkを`GGML_CUDA=ON`、
  `CMAKE_CUDA_ARCHITECTURES=70`で通常のフルビルドを行う。
- マスク版のバイナリ・CUDAライブラリ・起動スクリプトは保持している。
  旧サービス設定は`/mnt/data2/qsa27b/backups/native-mask-v1/`。

サービスの`EnvironmentFile`は`/mnt/data2/qsa27b/runtime-v2.env`を任意で読む。
A/B測定では`LLAMA_QSA_DENSE=1`で本体dense、`QSA_SPEC_TYPE=none`でMTPなしを選択できる。
通常運用へ戻す際はこれらのベンチマーク指定を外す。
`LLAMA_MTP_DENSE_BLOCK_MIN=-1`はMTPの全ブロックカーネルを無効化して通常FAへ戻す。
既定65,536。`QSA_DRAFT_MAX`でMTP提案長を変更できる。
