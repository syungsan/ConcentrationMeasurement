# トラブルシューティング

## `python`または`pip3.exe`を実行できない

PATHへ依存せず、同梱Pythonを明示して実行します。

```powershell
.\WPy64-312101\python\python.exe -m pip install -r requirements.txt
```

構文確認やテストにも同じ実行ファイルを使います。

```powershell
.\WPy64-312101\python\python.exe -m py_compile launcher.py
.\WPy64-312101\python\python.exe -m unittest discover -s tests
```

## ランチャーが起動しない

次を確認してください。

- `WPy64-312101/python/pythonw.exe`が存在する
- `launcher.py`がプロジェクト直下にある
- パッケージ内のフォルダ構成を変更していない
- エラー確認時は`pythonw.exe`ではなく`python.exe launcher.py`で起動する

## proxy動画または人物IDが表示されない

- datasetに`videos/proxy.mp4`があるか確認する。
- ID表示には`videos/proxy_ids.mp4`を使用する。
- `proxy_ids.mp4`を再生成する場合は`build_proxy_id_video.py`を実行する。
- ffmpegが`ffmpeg/bin`またはPATH上にあるか確認する。

## crop画像が表示されない

- `scripts/04_extract_frames_assets.py`が完了しているか確認する。
- `segment_frames.crop_path`と実ファイルの場所を確認する。
- datasetを移動した場合、絶対パスが古い場所を指していないか確認する。
- GUIのサムネ欠損デバッグ表示を有効にして探索先を確認する。

## 評価者DBをマージできない

よくある原因:

- 異なるdatasetのDBを混ぜている
- 同じ評価者名・同じsegmentに異なるスコアがある
- 採点GUIを終了する前にDBをコピーした
- 回収したDBを別の評価者のファイルで上書きした

評価者ごとにファイル名を変え、元のDBを保管したままマージしてください。

## Transformer学習でNaNが発生する

学習コードは非有限のlogit、予測、loss、gradientを検出し、不正なbatch更新やbest modelの破損を防ぎます。ログの次の項目を確認します。

```text
train_skipped_batches
train_nonfinite_pred
train_nonfinite_loss
train_nonfinite_grad
val_nonfinite_pred
```

対処順:

1. 入力特徴量にNaN / infがないか確認する。
2. learning rateを`1e-4`、必要なら`5e-5`へ下げる。
3. batch sizeを下げる。
4. Transformerの層数やhidden sizeを小さくする。
5. 同じ分割でGRUと比較する。

## 注釈動画に音声がない

OpenCVで映像を生成した後、ffmpegで元音声を結合します。元動画に音声がない、ffmpegが見つからない、または結合に失敗した場合は無音版が残ります。ffmpegの場所とコマンド出力を確認してください。

## Excelで`#NAME?`が表示される

古いExcelでは新形式関数を認識しない場合があります。[METRICS_GUIDE.md](METRICS_GUIDE.md)の互換関数を使用し、Excelでブックを開いて再計算してください。

## GPUを認識しない・処理が遅い

```powershell
.\WPy64-312101\python\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

`False`の場合はGPUドライバとPyTorchのCUDA版を確認します。負荷を下げる場合は、サンプリングfps、検出画像サイズ、姿勢推定画像サイズ、同時描画人数を減らしてください。
