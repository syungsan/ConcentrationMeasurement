# Current Task Summary 3: Feature/Temporal Model Evaluation Workflow

## 目的

集中度推定モデルについて、特徴量の組み合わせと時系列モデルの違いが精度に与える影響を検証できるようにする。

主な比較軸は以下。

- 特徴量: `image`, `skeleton`, `fusion`
- 時系列モデル: `gru`, `transformer`
- 評価方法: 回帰指標と、予測値を `1-7` に丸めた順序付き分類指標

想定する実験は `image/gru`, `image/transformer`, `skeleton/gru`, `skeleton/transformer`, `fusion/gru`, `fusion/transformer` の6条件比較。

## 実装した評価基盤

`scripts/07_train.py` に学習履歴の保存とグラフ出力を追加した。

学習後、既定では `<save_name stem>_training_report` に以下を出力する。

- `history.json`
- `concentration_<mode>_history.csv`
- `concentration_<mode>_loss.png` / `.svg`
- `concentration_<mode>_error.png` / `.svg`
- `concentration_<mode>_ordered_metrics.png` / `.svg`
- `situation_<mode>_history.csv`
- `situation_<mode>_loss.png` / `.svg`
- `situation_<mode>_accuracy.png` / `.svg`

出力先は `--report_dir` で変更できる。

## 独立テスト評価

`../scripts/tools/evaluate_labeled_dataset.py` を、学習済みcheckpointとラベル付き `dataset.sqlite` から直接評価できる独立スクリプトとして使う方針。

手作業で `model_vs_human.csv` を作らず、DB内の `labels` または `label_consensus` を正解として読み、モデル推論結果と突き合わせる。

主な出力。

- `predictions.csv`
- `metrics.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix_row_normalized.csv`
- `confusion_matrix.png` / `.svg`
- `confusion_matrix_row_normalized.png` / `.svg`
- `true_vs_predicted.png` / `.svg`
- `error_distribution.png` / `.svg`
- `run_config.json`

## Transformer学習のNaN対策

`transformer + fusion` 学習で途中から予測がNaNになる現象が発生した。

ログ上はep011までは改善し、ep012でvalidation予測が全NaN化していた。保存されたbest modelはep011以前の有限な重みと考えられるが、実験比較としては不安定なため、`scripts/07_train.py` に以下のガードを追加した。

- train中のlogits / prediction / lossがNaNまたはinfなら、そのbatchは更新しない
- gradient normがNaNまたはinfなら、`optimizer.step()` しない
- validation metricが非有限になったらbest重みに戻し、そのrunを停止
- historyに `train_skipped_batches`, `train_nonfinite_pred`, `train_nonfinite_loss`, `train_nonfinite_grad` を記録

Transformerではまず `--lr 1e-4`、不安定なら `--lr 5e-5` を試す方針。

## 配布パッケージ関連

評価者向け配布パッケージの評価順について、ランダムではなく時系列順にするよう変更した。

対象は `scripts/lib/research_schema.py` の `ensure_blind_assignments()`。

新しい順序。

1. `video_id`
2. `window.t_start`
3. `window.t_end`
4. `track_id`
5. `segment_id`

`blind_code` は匿名コードとして残すが、`display_order` のランダムシャッフルは廃止した。

また、ランチャーから作成される配布フォルダ名は `concentration_labeler_package` ではなく `concentration_labeler` に変更した。

## 注意点

指定されたvenvは `C:\venvs\concentration_measurement\Scripts\python.exe` だが、こちらの実行環境ではベースPython参照の問題で起動できないことがあった。そのため、一部の構文チェックや単体テストは同梱Python `WPy64-312101\python\python.exe` で確認している。

## 次にやるとよいこと

1. 同一train/validation/test分割で6条件を学習する。
2. 各checkpointを `../scripts/tools/evaluate_labeled_dataset.py` で同じtest datasetに評価する。
3. `metrics.json` を横並びに集計し、特徴量と時系列モデルの効果を比較する。
4. Transformer条件は低めの学習率から試し、NaNガードのログを確認する。


