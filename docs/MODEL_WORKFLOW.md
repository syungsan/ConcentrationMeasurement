# 学習・評価・推論

## モデル入力

| モード | 入力 |
|---|---|
| `image` | 頭～上半身のRGB画像 |
| `skeleton` | 姿勢キーポイント |
| `fusion` | 姿勢を主入力、RGBを補助入力として併用 |

標準構成は5秒間を2 fpsでサンプリングするため、1人・1Windowあたり10サンプルです。推論では5秒窓を1秒ずつずらし、約1秒ごとに集中度を更新します。

状況確率を集中度推定へ入力するかは、各スクリプトの`--use-situation-feature` / `--no-situation-feature`で切り替えられます。この設定はチェックポイントへ保存されます。

## 学習

合意ラベルを使う基本例:

```powershell
.\WPy64-312101\python\python.exe scripts\07_train.py `
  --data_roots datasets\lesson_001 `
  --db_paths merged\lesson_001_merged.sqlite `
  --label_source consensus `
  --min_raters 2 `
  --mode skeleton `
  --temporal gru `
  --objective ordinal `
  --save_name models\lesson_001.pt
```

個別評価者のラベルを使う場合:

```powershell
--label_source individual --raters evaluator01,evaluator02 --agg mean
```

主な比較軸:

- 入力: `image` / `skeleton` / `fusion`
- 時系列モデル: `gru` / `transformer`
- 目的関数: `ordinal` / `regression`
- 状況特徴量: あり / なし

`fusion`では`--fusion_image_scale`でRGB特徴の寄与を調整できます。未知環境への一般化では、服装や背景へ依存しにくい`skeleton`を基準に比較してください。

## データ分割

重複するWindowが学習側と検証側へまたがると、性能を過大評価します。最終評価では`window`より大きい単位を使用します。

| `split_unit` | 用途 |
|---|---|
| `window` | 開発初期の動作確認 |
| `dataset` | 未知動画への一般化確認 |
| `session` | 別授業・撮影セッションへの一般化確認 |
| `school` | 別学校への一般化確認 |

特定グループを検証用に固定する場合は`--holdout_groups`を使います。本試験の`pre` / `post`データを学習へ混ぜないでください。

## 学習レポート

既定ではチェックポイント名に対応する`*_training_report`へ履歴とグラフを保存します。出力先は`--report_dir`で変更できます。

- `history.json`
- 集中度・状況分類の履歴CSV
- loss、誤差、順序指標、accuracyのPNG / SVG

## ラベル付きデータによる独立評価

```powershell
.\WPy64-312101\python\python.exe scripts\tools\evaluate_labeled_dataset.py `
  --model models\lesson_001.pt `
  --data_roots datasets\test_lesson_001 `
  --label_source consensus `
  --mode skeleton `
  --output-dir outputs\verifications\test_eval
```

個別評価者を正解とする場合:

```powershell
--label_source individual --raters evaluator01,evaluator02 --agg mean
```

状況の与え方は`--situation-source true`（DBの正解状況）または`predicted`（チェックポイントの状況分類器）を指定します。比較実験ではデータ分割、seed、状況の与え方、除外条件を固定してください。

主な出力:

- `predictions.csv`
- `metrics.json`
- `classification_report.csv`
- confusion matrix（CSV / PNG / SVG）
- true vs predicted、誤差分布（PNG / SVG）
- `run_config.json`

指標の解釈は[METRICS_GUIDE.md](METRICS_GUIDE.md)を参照してください。

## オフライン推論

```powershell
.\WPy64-312101\python\python.exe scripts\09_video_offline.py `
  --ckpt models\lesson_001.pt `
  --mode skeleton `
  --data_root datasets\unseen_lesson `
  --annotate_out outputs\annotated.mp4 `
  --log_db outputs\pred_log.sqlite
```

5秒分のサンプルが揃う前は`warm-up n/10`と表示されます。これは集中度ではなく蓄積済みサンプル数です。注釈動画は可能な場合、ffmpegで元動画の音声を結合します。

## リアルタイム推論

```powershell
.\WPy64-312101\python\python.exe scripts\08_realtime.py `
  --ckpt models\lesson_001.pt `
  --mode skeleton `
  --source 0 `
  --show
```

処理負荷が高い場合は、sampling fps、検出・姿勢推定の画像サイズ、描画人数を下げます。

## 追加学習

新規データだけの追加学習は過去データへの性能低下（破滅的忘却）を起こす可能性があります。

1. 新規データだけで改善可能性を短時間確認する。
2. 新旧それぞれの検証セットで評価する。
3. 実運用候補は過去データの一部を混ぜて学習する。
4. 節目では全学習データを統合して再学習する。
5. 最終評価データは常に分離する。
