# 自分のデータで学び直す

同梱のモデルはゲームの絵だけで学習しています。実際のカメラ・道で使うなら、その環境で走った記録で学び直してください。
**人手のラベルは要りません**: PathNet の教師は「その後に実際に通った線」を位置の記録から自動で作ります。

準備(学び直しの道具は torch を使う。数 GB):

```bash
bash setup.sh --train        # Windows: setup.bat --train
```

## 1. 記録の形(1 走行 = 1 フォルダ)

```
my_drive1/
  frames/000000.jpg …         絵(名前の順 = 時刻の順。jpg / png)
  poses.csv                   1 枚 1 行の位置・向き
  camera.json                 カメラの設定(configs の "camera" と同じ形。無ければ --config で渡す)
```

`poses.csv`(見出しの行あり):

```
frame,t,x,y,z,yaw_deg,pitch_deg,roll_deg
0,0.000,0.000,0.000,0.000,42.55,-4.59,3.22
1,0.269,2.477,2.175,-0.226,39.97,-3.92,1.94
```

| 列 | 意味 |
|---|---|
| `frame` | 絵の番号(frames/ を名前の順に並べたときの何番目か。0 から) |
| `t` | 時刻 [s] |
| `x`, `y`, `z` | 位置 [m]。**右手系・z が上**(x 東・y 北 など。原点はどこでもよい)。車の基準点(camera.json の fwd_m・right_m を測った点)の位置 |
| `yaw_deg` | 向き: +x 軸から反時計回り(上から見て左回り)が正 |
| `pitch_deg`, `roll_deg` | 機首上げが正・右側が下がるのが正(無ければ 0 とみなす) |

向きは `fwd_x, fwd_y, fwd_z, right_x, right_y, right_z`(前と右の単位ベクトル)の列で書いてもかまいません。
`samples/*/` が実例です(ゲームのテレメトリから書き出したもの)。

**位置の取り方の例**:
- GNSS(RTK が望ましい)+ IMU: 緯度経度を原点まわりの平面の m に直す(x = (経度 − 経度₀)·cos(緯度₀)·111,320、y = (緯度 − 緯度₀)·110,540)。
  向きは IMU、または前後の位置の差から
- 車輪のオドメトリ・Visual SLAM・LiDAR SLAM(ROS の `/odom` など)
- シミュレータ(CARLA などの車の位置・向き)
- 絵の時刻に合わせて位置を補間する(ずれると教師がずれる)

**走り方**: 教師は「実際に通った線」なので、**走ってほしい通りに走った記録** を使います(車線を守る・障害物を避ける)。
止まっている所・下がっている所は自動で除きます。いろいろな明るさ・天気・道を混ぜてください(元の開発では 30 万枚以上)。

## 2. 当たり具合を測る → 学び直す → もう一度測る

```bash
# 今のモデルがどれくらい当たるか(学習に使わない走行で)
.venv/bin/python tools/eval_pathnet.py my_drive_test --config configs/my_car.json

# 学び直す(同梱のモデルの続きから。各走行の終わり 15% を検証に回す)
.venv/bin/python tools/train_pathnet.py my_drive1 my_drive2 my_drive3 --config configs/my_car.json \
    --out models/pathnet_my.onnx --epochs 6

# 学び直したモデルで測る
.venv/bin/python tools/eval_pathnet.py my_drive_test --config configs/my_car.json --model models/pathnet_my.onnx
```

良くなっていれば、設定の `"models": {"pathnet": "models/pathnet_my.onnx"}` に替えて使います。

- `ずれ` = 道なり 8〜22m 先の 4 か所で、予想と実際に通った点の距離の平均。元の開発では学習したコースで約 0.8m
- 学び直しの最後に、検証で **自信の物差し**(`sd_scale`)を合わせて JSON に書きます。自信 0.5 以上の絵のずれが 0.66m(基準の世界)
  ほどになるように合わせる(`--trust-err`)
- データが数万枚以上あれば `--scratch`(一から)も試す価値がある。少ないうちは続きから(既定)の方が安定
- **学習に使った走行で測らない**(良く見えすぎる)。元の開発でも「学習していないステージ」で比べてから入れ替えた

## 3. 障害物の検出を学び直す

```
my_set/
  frames/xxx.jpg
  masks/xxx.png          同じ名前の塗り(0 以外 = 当たる物)。塗りの無い絵は「何も無い」として学ぶ
  camera.json
```

```bash
.venv/bin/python tools/train_obstacle.py my_set1 my_set2 --config configs/my_car.json --out models/obstacle_my.onnx
```

- 塗りは手で(CVAT・Label Studio などで PNG に書き出す)、またはほかのセグメンテーションの出力で作れます
- **何も無い絵も混ぜる**。検証の「何も無い絵」で鳴る割合が 3% 以下になるしきい値を選んで JSON に書きます
- 元の開発では手で塗らず、走行の記録から教師を自動で作りました: 実際に当たって止まった所(下がった・復帰した所)を、その 4.5〜25m 手前の絵に
  写した箱を「物」、その後に実際に通った車幅の帯を「空き」、それ以外を「分からない」とした。実車ではバンパーのセンサー・人の介入
  (ブレーキを踏んだ所)などで同じことができます

## 4. うまくいかないとき

- `教師のある絵が少なすぎる` → poses.csv の frame と絵の枚数が合っているか、位置が 65m 先まで続いているか
- ずれが大きいまま → カメラの設定(とくに pitch_deg・height_m)を `tools/check_camera.py` で確かめる。位置と絵の時刻のずれ
- 自信がいつも低い/高い → `--trust-err` を変えるか、検証の走行を増やす
