# Concentration Measurement (Offline)

介入研究向けのデータ区分、盲検採点、学校単位検証、教室集約、療法前後監査については [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md) を参照してください。

## 1) Setup
- Python 3.10+ 推奨
- ffmpeg をインストール（proxy作成に使用）

```bash
pip install -r requirements.txt
```

## Labeling workflow

通常はプロジェクト直下のランチャーを起動します。

```bash
WPy64-312101/python/python.exe launcher.py
```

1. 代表者が共有状況を設定し、全区間をロックします。

```bash
python scripts/06_label_gui.py --dataset datasets/lesson_001 --task situation
```

2. 各評価者が同じDBに集中度を入力します。

```bash
python scripts/06_label_gui.py --dataset datasets/lesson_001 --task rating
```

## Collecting evaluator databases

- 評価者は採点GUIを終了してから `db/dataset.sqlite` を管理者へ渡します。
- 管理者はランチャーで対象datasetを選択し、「管理」→「評価者DBをマージ」を実行します。
- マージ後DBには評価者別の `labels` と平均値の `label_consensus` が作成されます。
- 同名評価者の矛盾するラベルや、異なるdatasetのDBはエラーとなります。

平均ラベルで学習する例：

```bash
python scripts/07_train.py \
  --data_roots datasets/lesson_001 \
  --db_paths merged/lesson_001_merged.sqlite \
  --label_source consensus \
  --mode fusion \
  --fusion_image_scale 0.25
```

`fusion` はskeletonを主入力、RGBを補助入力として扱います。RGBモデル入力は既定で頭〜上半身ROIに切られ、保存cropは採点確認用に全身のまま残ります。

## Proxy video with subtle track IDs

`02_detect_track.py` 実行後、`videos/proxy_ids.mp4` が自動生成されます。`06_label_gui.py` はこのファイルがあれば通常の `proxy.mp4` より優先して再生します。再生中にDBへ問い合わせないため、人物ID表示によるカクつきを避けられます。

既存の `detections` からID付きproxyだけ作り直す場合：

```bash
python scripts/tools/build_proxy_id_video.py --dataset datasets/lesson_001 --label-fps 4
```

## Building evaluator packages

共有状況を全区間で確定・ロックした後、管理者版ランチャーで
配布したいdatasetをCtrlまたはShiftで複数選択し、
「管理」→「評価者用パッケージを作成」から生成できます。

CLIでの例：

```bash
WPy64-312101/python/python.exe scripts/tools/build_labeling_package.py \
  --datasets datasets/lesson_001 datasets/lesson_002 \
  --output dist/concentration_labeler
```

生成される評価者版は、管理者モードとDBマージメニューが非表示になり、
評価者名は配布先で初回起動時に入力します。採点後はパッケージ内の
`datasets/<dataset>/db/dataset.sqlite` を回収します。
