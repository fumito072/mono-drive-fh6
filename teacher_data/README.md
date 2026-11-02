# FH6 人間プレイの教師データ

2026年11月2日時点で取り込み・同期検査を終えた **66エピソード、258,042枚** を公開しています。画像と位置・操作量の組を学習に使えます。約31.5GiBの実データは [GitHub Release](https://github.com/fumito072/mono-drive-fh6/releases/tag/human-data-2026-11-02) に分けて置き、Git本体には [一覧とSHA-256](catalog.json)、取得プログラム、説明を置いています。

## 内容と区分

- 学習用: 443区間・163,318枚
- 検証用: 290区間・94,724枚
- 同期検査を通らず保留中の2件は含めていません。

録画ごとに時間を分けた元の区分をそのまま保存しています。同じ録画の隣接フレームをランダムに混ぜ直さないでください。録画をまたいだ未知環境での性能は、この区分だけでは保証できません。

各ZIPは独立したエピソードです。展開後の構成は次のとおりです。

```text
human/episode-<公開ID>/
  episode.json
  train_000/             # 記録によっては train_00
    frames/0000000.jpg
    poses.csv
    actions.csv
    camera.json
    provenance.json
  validation_000/
    ...
```

画像は取り込み済みJPEGをそのまま収録しています。元動画の全フレームではなく、学習に使う抽出済み画像です。`poses.csv` と `actions.csv` の `frame` は、区間内の画像をファイル名順に並べた0始まりの番号に対応します。

## ダウンロード

リポジトリのルートで実行します。追加パッケージは不要です。

```powershell
python tools/download_human_dataset.py --output datasets/public-human-20261102
```

プログラムはZIPのサイズ・SHA-256を照合してから展開し、`training-manifest.json` を作ります。ZIPを再ダウンロードせず再実行できるよう、取得済みZIPは出力先の `.archives` に残します。全件ではZIPと展開先を合わせて約64GiBの空き容量が必要です。

小さく試す場合は一覧からIDを選んでください。

```powershell
python tools/download_human_dataset.py --episodes episode-047c5a622b118f6c --output datasets/human-sample
```

## 学習に使う

PyTorch、ONNXなどの学習環境を用意した後、取得プログラムが出したマニフェストを指定します。次の例は初期重みなしで学習します。

```powershell
python tools/train_pathnet.py --dataset-manifest datasets/public-human-20261102/training-manifest.json --scratch --out models/pathnet_human.onnx --device cuda
```

将来の軌跡ラベルは位置・向きの時系列から作ります。停止・後退や十分な先の軌跡がないフレームなどを学習器が除くため、実際の学習サンプル数は公開画像数より少なくなります。学習器は学習用区間の一部も内部検証に回すため、その検証枚数は上記の専用検証用94,724枚より多くなる場合があります。

直線に偏った記録から曲がる場面を重点的に学習するには、学習コマンドに `--balance-maneuvers --curve-weight 2 --epochs 8` を追加できます。学習用の目標軌跡だけを使って出現頻度を調整し、検証用画像は増幅しません。学習器は最終回だけでなく、検証誤差が最も小さかった回の重みを保存します。条件を増やしても実走性能が改善するとは限らないため、独立した検証と実走評価を行ってください。

復元した教師データの速度・曲がり方・同期情報を再集計する例です。

```powershell
python tools/analyze_human_curriculum.py --manifest teacher_data/catalog.json --data-root datasets/public-human-20261102
```

`poses.csv` は `frame,t,x,y,z,yaw_deg,pitch_deg,kmh` を持ちます。位置はゲーム内のメートル座標を右手系・z上向きに変換した値、角度は度、速度はkm/hです。カメラの位置・画角は初期推定値であり、精密な実測校正ではありません。

`actions.csv` は記録方式に応じて2種類あります。

- 共通: `frame,video_pts_s,applied_steer`
- 新方式: `telemetry_seq,age_ms,throttle,brake`
- 旧方式: `csv_elapsed_s,applied_throttle,applied_brake`

アクセル・ブレーキは0〜1、操舵はおおむね−1〜1です。値は **ゲームのData Outが報告した状態** であり、物理コントローラーの生入力ではありません。新方式も受信時刻と描画・ゲーム内部の遅延は別です。旧方式は同期オフセットの推定を含み、`provenance.json` の `offset_verified_for_actions=false` を維持しています。これらを無条件に正確な操作指令として模倣学習しないでください。

## 出どころと公開範囲

利用者が用意したForza Horizon 6の人間プレイ記録から生成しました。ファイル名の個人名・PCの絶対パス・元動画の保存先・Driveアカウントは公開用IDと許可した項目に置き換えています。元画像そのものは加工していないため、ゲーム内のHUDやプレイヤー名が映る場合があります。画像の完全な匿名化を主張するものではありません。

元動画、未検証データ、自動走行の記録、認証情報、学習済みモデルはこのデータセットに含めていません。ゲーム画像など第三者の権利は各権利者に帰属します。データに新しい再配布ライセンスは付与していません。

公開用パッケージを手元の準備済みデータから作る処理は `tools/package_human_dataset.py` です。この処理は元データを変更せず、ネットワークへのアップロードも行いません。公開は明示的に行ってください。
