# 公開モデルの重み

[モデル重みのリリース](https://github.com/fumito072/mono-drive-fh6/releases/tag/model-weights-2026-11-02)に、このプロジェクトで保存していた完成済みの重み **132ファイル** を収録しています。[catalog.json](catalog.json) にファイル名、SHA-256、構成、利用条件、走行モデルの状態を記録しています。

自作のPathNet・障害物モデル、過去の全候補モデル、Depth Anything V2 Small、YOLO11s・YOLOE、使用していたSegFormer b0/b2、Qwen3.5-4B、Qwen-Drive-1.0-4BとVLM用構成を含みます。認識用の補助配列、設定、トークナイザーも含めています。未完了のダウンロード断片4個は、利用可能な重みではないため除いています。Qwen3-VL-4Bのローカル保存は未完了でした。

## 取得と復元

全体のダウンロード量は約18.2GiBです。重複したQwen-Driveの重みを一度だけ配布し、大きなファイルは1.5GiB以下の分割ファイルにしています。取得ツールが結合後のSHA-256も検査し、元のフォルダー構成に戻します。キャッシュと展開先を含め、60GiB程度の空き容量を用意してください。

```powershell
python tools/download_model_weights.py --output .
```

まず走行・物体認識モデルだけを取得する場合は次を使います。この場合、Qwenの大きな重みは取得しません。

```powershell
python tools/download_model_weights.py --small-only --output .
```

既存の重みと内容が異なる場合は自動上書きしません。意図的に置き換えるときだけ `--overwrite` を付けてください。取得だけでは、走行中のモデル選択やゲーム設定は変わりません。ゲーム固有の画面校正、仮想コントローラー、依存ライブラリは別途用意してください。

## 走行モデルの状態

公開は性能検証への合格を意味しません。

追加の時系列操舵モデルと検証結果は [2026-11-03の実験リリース](https://github.com/fumito072/mono-drive-fh6/releases/tag/temporal-steering-2026-11-03) にあります。初回候補は全体精度の基準を満たさず不採用です。重みの公開と運転への採用は別に扱います。

- `active_runtime_baseline_at_snapshot`: 書き出し時に実行設定が指していたモデル。
- `live_trial_rolled_back`: 実走で問題があり切り戻したモデル。通常の走行用に選ばないでください。
- `offline_evaluation_in_progress_at_snapshot`: 書き出し時に比較検証中だった最新候補。
- `archived_candidate_not_auto_selected`: 過去候補の保存。採用済みという意味ではありません。
- `available_not_a_driving_quality_guarantee`: 配布元のモデルや基礎モデルなど。

左車線中央を60km/hで安定して走る目標は未達です。新しいファイル名だけを基準に選ばず、個別の検証結果を確認してください。

## 出どころと利用条件

- 自作のPathNet・障害物モデル: 本プロジェクトで学習。新しい再配布ライセンスは付与していません。基礎モデルにはDiRT Rally 2.0、候補にはFH6のゲーム画像由来の学習が含まれます。
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) / [Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B): Apache-2.0。重みは配布元のものです。Qwen-DriveのVLM用構成はローカルで設定とキー対応を変更していますが、重みは同一です。
- [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2): Apache-2.0。ローカルでONNXに変換したものです。
- [SegFormer](https://github.com/NVlabs/SegFormer): NVIDIAのライセンスにより、非商用の研究・評価用途に限定されます。[ライセンス全文](licenses/SegFormer-NVIDIA.txt)を同梱しています。
- [Ultralytics](https://github.com/ultralytics/ultralytics/tree/v8.4.189): AGPL-3.0。[ライセンス全文](licenses/Ultralytics-AGPL-3.0.txt)とバージョン8.4.189のソースアーカイブを重みと同じZIPに同梱しています。`fh6-yoloe-26s.pt` はローカルのプロンプト設定を保存した派生チェックポイントです。

`runtime-models.zip` 内の `model-licenses/` にライセンスと出典を収録しています。各配布元の権利表示・条件が適用され、すべてを同じ条件で扱えるわけではありません。

公開時にモデルの付帯メタデータから個人名を含むデータ名やPCのパスを取り除きました。YOLOEの派生チェックポイント内のPCパスもファイル名だけに置き換え、保存されたテンソル本体が変わっていないことを確認しています。重みの精度向上を意味する変更ではありません。
