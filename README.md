# Concentration Measurement (Offline)

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
python scripts/05_label_gui.py --dataset datasets/lesson_001 --task situation
```

2. 各評価者が同じDBに集中度を入力します。

```bash
python scripts/05_label_gui.py --dataset datasets/lesson_001 --task rating
```

## Collecting evaluator databases

- 評価者は採点GUIを終了してから `db/dataset.sqlite` を管理者へ渡します。
- 管理者はランチャーで対象datasetを選択し、「管理」→「評価者DBをマージ」を実行します。
- マージ後DBには評価者別の `labels` と平均値の `label_consensus` が作成されます。
- 同名評価者の矛盾するラベルや、異なるdatasetのDBはエラーとなります。

平均ラベルで学習する例：

```bash
python scripts/06_train.py \
  --data_roots datasets/lesson_001 \
  --db_paths merged/lesson_001_merged.sqlite \
  --label_source consensus \
  --mode fusion
```

## Building evaluator packages

共有状況を全区間で確定・ロックした後、管理者版ランチャーで
配布したいdatasetをCtrlまたはShiftで複数選択し、
「管理」→「評価者用パッケージを作成」から生成できます。

CLIでの例：

```bash
WPy64-312101/python/python.exe scripts/tools/build_labeling_package.py \
  --datasets datasets/lesson_001 datasets/lesson_002 \
  --output dist/concentration_labeler_teacherA \
  --evaluator-name teacherA
```

生成される評価者版は、管理者モードとDBマージメニューが非表示になり、
評価者名も変更できません。採点後はパッケージ内の
`datasets/<dataset>/db/dataset.sqlite` を回収します。
