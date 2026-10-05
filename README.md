# Concentration Measurement

授業動画から集中度を推定・分析するPythonプロジェクトです。このREADMEでは、開発環境の導入方法だけを説明します。操作方法や分析手順は[`docs`](docs/)を参照してください。

## 動作環境

- Windows 10 / 11 または Apple Silicon Mac（M1以降）
- Python 3.12（同梱WinPythonを推奨）
- NVIDIA GPU（学習・推論を高速化する場合）
- ffmpeg（動画の変換・音声結合に使用。リポジトリの`ffmpeg/bin`も利用可能）

## Mac（Mシリーズ）で実行する方法

arm64版Python 3.12の仮想環境を使用してください。このMacで確認した環境は
`/Users/syungsan/venvs/3.12/concentration_measurement/` です。

```bash
source /Users/syungsan/venvs/3.12/concentration_measurement/bin/activate
python -m pip install -r requirements_for_mac.txt
brew install ffmpeg-full
python launcher.py
```

`brew install ffmpeg-full` はffmpegが未導入の場合に実行します。ランチャーは起動に使ったPythonを
子プロセスにも使用します。Tkinterが必要です（`python -m tkinter` で確認できます）。
HomebrewのPythonでTkinterがない場合は対応する `python-tk@3.12` を導入してください。

学習・検出・姿勢推定・動画推論はCUDA → MPS → CPUの順で利用可能なデバイスを自動選択します。
MPSはApple SiliconのGPUを使うPyTorchバックエンドです。
学習・リアルタイム推論・オフライン推論・評価には `--device mps` または `--device cpu` を指定できます。
前処理の検出と姿勢推定は `config.yaml` の `detection_tracking.device` / `pose.device` で切り替えます。

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
python scripts/07_train.py --help
python scripts/09_video_offline.py --help
python scripts/tools/build_face_db.py --device auto
```

InsightFace（顔DB作成・顔画像収集・除外判定）はONNX Runtimeを使います。
Macの `auto` はCPUですが、`coreml` を指定するとCoreML経由でGPUを利用できます。
顔画像収集GUIでは「Mac GPU（CoreML）」を選択してください。
顔DB作成では `--device coreml`、除外判定では `config.yaml` の
`face_exclusion.device: "coreml"` を指定します。
CoreML非対応の演算はCPUで実行されます。初回のモデル準備には時間がかかることがあります。
`mps` はPyTorch用の指定で、顔認識部分ではCPU扱いです。
Mac用依存関係には `onnxruntime` を使用し、`onnxruntime-gpu` はインストールしません。
顔画像収集は顔検出のみ実行し、特徴抽出・ランドマーク推定・年齢性別推定は省略します。

このMacのダミー画像での5回の推論中央値（初期化・ウォームアップ除外）は、
CPU / CoreMLで顔検出 94.6 / 19.0 ms、特徴抽出 52.9 / 10.9 msでした。
実際のカメラ映像のFPSを保証する測定ではありません。
[CoreMLの公式説明](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)

モデルファイルとデータセットは別途必要です。`config.yaml` のモデル配置先を確認してください。
MacでGPUメモリが不足する場合は学習の `--batch_size` を小さくするか `--device cpu` を指定します。
MPS未対応演算のエラーが出る場合も `--device cpu` で実行できます。

## 同梱WinPythonを使う方法

リポジトリに`WPy64-312101`が含まれている場合、追加のPython環境を作成せずに利用できます。

```powershell
.\WPy64-312101\python\python.exe --version
.\WPy64-312101\python\python.exe -m pip install -r requirements.txt
```

PyTorchをGPUで使用する場合は、利用するCUDA環境に合うパッケージをインストールします。CUDA 12.8向けの例:

```powershell
.\WPy64-312101\python\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 仮想環境を作る方法

システムのPython 3.12を使う場合:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorchはCPU版または使用するCUDA環境に合う版を別途インストールしてください。

## Ubuntu / LinuxでCUDAを使う場合

仮想環境を作成し、依存関係とCUDA対応PyTorchをインストールします。CUDA 12.8向けの例:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`config.yaml`の`device: "auto"`と、各コマンドの`--device auto`では、OSに関係なくCUDAを自動判定します。選択順はCUDA、macOSのMPS、CPUです。顔認証はONNX Runtimeの`CUDAExecutionProvider`を独立して判定します。

CUDAの認識状態は次のコマンドで確認できます。

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 導入確認

主要モジュールを読み込めることを確認します。

```powershell
.\WPy64-312101\python\python.exe -c "import cv2, numpy, pandas, PySide6, torch, ultralytics; print('OK')"
```

ラベラーのランチャーを起動できれば導入完了です。

```powershell
.\launch_labeler.bat
```

## ドキュメント

- [ラベリングとデータセット運用](docs/LABELING_GUIDE.md)
- [学習・評価・推論](docs/MODEL_WORKFLOW.md)
- [研究設計・集計・AB解析](docs/RESEARCH_ANALYSIS.md)
- [評価指標](docs/METRICS_GUIDE.md)
- [トラブルシューティング](docs/TROUBLESHOOTING.md)

## リアルタイム推論の人物検出モデルを切り替える

通常の `08_realtime.py` のコマンドに `--light-det` を追加すると、人物検出を軽量な `models/yolov8n.pt` に切り替えます。省略時は `config.yaml` の `detection_tracking.model` を使用します。

```bash
python scripts/08_realtime.py --ckpt models/your_model.pt --show --light-det
```

任意の重みは `--det-model models/your_detector.pt` で指定できます（相対パスはプロジェクト基準、絶対パスも使用可能）。`--light-det` と `--det-model` は同時には指定できません。姿勢推定モデルと集中度推定モデルはこのオプションでは変更されません。

## リアルタイム推論で顔認識による除外を省く

通常の `08_realtime.py` のコマンドに `--no-face-exclusion` を追加します。
`config.yaml` の `face_exclusion.enabled` より優先され、顔DBと顔認識モデルを
読み込まず、顔照合による人物の除外を行いません。引数を省略すると従来どおり設定に従います。

```bash
python scripts/08_realtime.py --ckpt models/your_model.pt --show --no-face-exclusion
```

`models/your_model.pt` は学習済みチェックポイントのパスに置き換えてください。

`04_extract_frames_assets.py` でも同じ引数が使えます。
既存の除外判定は変更せず、顔認識による新規判定・更新を省きます。
クロップ画像と姿勢データの抽出は通常どおり行います。

```bash
python scripts/04_extract_frames_assets.py --data_root datasets/lesson_001 --no-face-exclusion
```

ID付き動画の生成には字幕フィルター（libass）が必要です。
Macでは通常版ffmpegに加えて `ffmpeg-full` を導入してください。
スクリプトがHomebrewの `ffmpeg-full` を自動検出するため、強制リンクは不要です。
検出結果が保存済みなら、次のコマンドでID付き動画だけ再生成できます。

```bash
python scripts/tools/build_proxy_id_video.py --dataset datasets/sample --label-fps 4
```
