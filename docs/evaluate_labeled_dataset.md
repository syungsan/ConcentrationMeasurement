# Labeled test dataset evaluation

学習済みcheckpointを、ラベル付きの `dataset.sqlite` に対して直接評価するスクリプトです。`model_vs_human.csv` を手作業で作る必要はありません。

## 実行例

```powershell
python.exe scripts\tools\evaluate_labeled_dataset.py `
  --model models\pilot_fusion.pt `
  --data_roots datasets\test_lesson_001 `
  --label_source consensus `
  --mode fusion `
  --output-dir outputs\verifications\test_eval
```

`--db_paths` を省略すると、各 `--data_roots` の下にある `../config.yaml` の `paths.db_path` が使われます。複数datasetを評価する場合はカンマ区切りで指定します。

```powershell
--data_roots datasets\test_a,datasets\test_b
```

## 個別評価者ラベルで評価する場合

```powershell
--label_source individual --raters evaluator01,evaluator02 --agg mean
```

## 状況ラベルの扱い

既定ではDB内の真の状況ラベルを使います。

```powershell
--situation-source true
```

checkpointに状況分類器が保存されている場合は、実運用に近い形で予測状況を使えます。

```powershell
--situation-source predicted
```

## 出力

- `predictions.csv`
  - `true_score`, `pred_score`, `true_class`, `pred_class`, `score_error`, `class_error`
- `metrics.json`
  - `mae`, `rmse`, `r2`, `pearson`, `spearman`
  - `exact_accuracy`, `within_1_accuracy`, `within_2_accuracy`
  - `weighted_precision`, `weighted_recall`, `weighted_f1`
  - `quadratic_weighted_kappa`
- `classification_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix_row_normalized.csv`
- `confusion_matrix.png`
- `confusion_matrix.svg`
- `confusion_matrix_row_normalized.png`
- `confusion_matrix_row_normalized.svg`
- `true_vs_predicted.png`
- `true_vs_predicted.svg`
- `error_distribution.png`
- `error_distribution.svg`
- `run_config.json`

グラフは確認しやすいPNGと、資料化・拡大表示しやすいSVGの両方を出力します。
