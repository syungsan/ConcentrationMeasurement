# Concentration Measurement

授業動画から集中度を推定・分析するPythonプロジェクトです。このREADMEでは、開発環境の導入方法だけを説明します。操作方法や分析手順は[`docs`](docs/)を参照してください。

## 動作環境

- Windows 10 / 11
- Python 3.12（同梱WinPythonを推奨）
- NVIDIA GPU（学習・推論を高速化する場合）
- ffmpeg（動画の変換・音声結合に使用。リポジトリの`ffmpeg/bin`も利用可能）

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
