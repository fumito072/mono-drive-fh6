# ライセンスと出どころ

| 物 | 出どころ | ライセンス・条件 |
|---|---|---|
| `mono_drive/`・`tools/`・`tests/`・`examples/`・文書 | このプロジェクトで書いたもの | 権利者(このプロジェクトの持ち主)が決める。外へ公開・配布する前に確認 |
| `models/pathnet.onnx`・`models/obstacle.onnx` | このプロジェクトで学習したもの。学習の絵はゲーム DiRT Rally 2.0 の画面 | 同上。ゲームの画面から学んだモデルなので、外へ公開する・売る前に、ゲームの利用規約に照らして確認 |
| `models/depth_anything_v2_small_266x420.onnx` | Depth Anything V2 Small(Lihe Yang ほか)。<https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf> を 266x420 の入力で ONNX に書き出したもの(重みは変えていない) | **Apache License 2.0**(全文は [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt))。Base・Large の版は非商用ライセンスなので取り替えるときは注意 |
| `samples/` | ゲーム DiRT Rally 2.0 の画面とテレメトリ(このプロジェクトで記録) | 動作確認用。ゲームの著作物なので、外へ公開するときは自分で撮った絵に差し替える |

入っていないもの:
- **YOLO11n(Ultralytics)と、それから作った道の塗り分け**: AGPL-3.0 のため入れていない。人・車の検出を足したいときは、ライセンスを確かめて
  自分で入れる(元の開発では車・人の検出に使っていた)

`setup` が入れるパッケージ(このキットには含めず、pip で入る): numpy(BSD-3-Clause)・opencv-python(Apache-2.0、同梱の FFmpeg は LGPL)・
onnxruntime(MIT)。学び直しで使う torch(BSD-3-Clause)・onnx(Apache-2.0)。
