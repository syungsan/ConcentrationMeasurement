# 現在のタスクまとめ

最終更新: 2026-07-17

この文書は、集中度推定システムの現在の改修状態と、介入プログラム「アクティ・ブレイク（AB）」を含む運用手順をまとめたものです。

## 目的

このプロジェクトでは、授業中の児童の集中度を、姿勢キーポイント、RGB画像、授業シチュエーションの時系列データから推定します。

現在の主目的は、児童個別の評価よりも、教室全体の集中度推移を見て、AB実施前後で集中度が改善するかを検証することです。

## 現在の基本方針

- 集中度ラベルは 1〜7 のみを使う。
- 観察不能フラグ、確信度、ノート欄、気温データは使わない。
- シチュエーションは `聞く` / `書く` / `話し合う` を基本とする。
- AB区間は、推論後に `AB` として後付けできる。
- 学習では `skeleton` を主軸にする。
- `fusion` ではRGBを補助情報として弱く使う。
- RGB入力は全身ではなく、頭〜上半身ROIを使う。
- 採点確認用のcropは全身のまま残す。

## 主要スクリプト

現在の番号付きメインスクリプトは以下の流れです。

| 番号 | スクリプト | 役割 |
|---|---|---|
| 01 | `scripts/01_make_proxy.py` | raw動画からproxy動画を作成 |
| 02 | `scripts/02_detect_track.py` | 人物検出・追跡、`detections` 作成、ID付きproxy生成 |
| 03 | `scripts/03_make_segments.py` | 時間窓と人物segment作成 |
| 04 | `scripts/04_extract_frames_assets.py` | cropとpose asset作成 |
| 05 | `scripts/05_configure_research_dataset.py` | 研究用メタデータ設定 |
| 06 | `scripts/06_label_gui.py` | シチュエーション設定・集中度採点 |
| 07 | `scripts/07_train.py` | 学習 |
| 08 | `scripts/08_realtime.py` | リアルタイム推論 |
| 09 | `scripts/09_video_offline.py` | 動画オフライン推論・注釈動画作成 |
| 10 | `scripts/10_aggregate_predictions.py` | 推論ログ集計 |
| 11 | `scripts/11_intervention_analysis.py` | 介入前後解析 |
| 12 | `scripts/12_measurement_invariance_audit.py` | 測定不変性監査 |

## ラベル作成GUI

`06_label_gui.py` は2つのモードで使います。

### シチュエーション設定

代表者が、授業動画の時間窓ごとにシチュエーションを設定します。

```powershell
.\WPy64-312101\python\python.exe scripts\06_label_gui.py `
  --dataset datasets\lesson_001 `
  --task situation `
  --name representative
```

シチュエーション設定後、評価者が集中度を採点します。

### 集中度採点

```powershell
.\WPy64-312101\python\python.exe scripts\06_label_gui.py `
  --dataset datasets\lesson_001 `
  --task rating `
  --name evaluator01
```

評価者は集中度 1〜7 だけを入力します。

現在は、30秒までの評価区間をまとめて表示し、採点結果を内部の5秒segmentへ展開します。右ペインのパラパラアニメは最大60枚、2fps相当です。ただし、実際の枚数は `segment_frames` に保存されているcrop枚数に依存します。

## ID付きproxy動画

`QVideoWidget` 上にリアルタイムでIDを重ねる方式は、環境によって表示されず、再生中にカクつく原因になりました。そのため現在は、IDを焼き込んだproxy動画を事前生成します。

`02_detect_track.py` 実行後、以下が自動生成されます。

```text
videos/proxy_ids.mp4
```

`06_label_gui.py` は `proxy_ids.mp4` が存在する場合、自動的に通常の `proxy.mp4` より優先して再生します。

既存の `detections` からID付きproxyだけ作り直す場合は、以下を実行します。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\build_proxy_id_video.py `
  --dataset datasets\lesson_001 `
  --label-fps 4
```

ID表示は、頭の少し上に薄く表示されます。

## 研究用データ区分

`05_configure_research_dataset.py` で、datasetに研究メタデータを付けます。

例:

```powershell
.\WPy64-312101\python\python.exe scripts\05_configure_research_dataset.py `
  datasets\lesson_001 `
  school_a `
  5 `
  5A `
  normal
```

主な区分は以下です。

| phase | 意味 | 学習での扱い |
---|---|---|
| `normal` | 通常授業・介入前・通常開発データ | 通常使用 |
| `pilot_post` | 開発用のパイロットAB後データ | `--include_pilot_post` 指定時のみ使用 |
| `pre` | 本試験の介入前 | 学習には使わず推論・検証用 |
| `post` | 本試験の介入後 | 学習には使わず推論・検証用 |

重要なのは、検証対象の本試験データを学習に混ぜないことです。開発用に別途用意した `pilot_post` は、必要に応じて学習へ入れて構いません。

## 学習

基本は `skeleton` を主軸にします。

```powershell
.\WPy64-312101\python\python.exe scripts\07_train.py `
  --data_roots datasets\normal_train `
  --label_source individual `
  --raters evaluator01 `
  --split_unit window `
  --mode skeleton `
  --save_name models\normal_only.pt
```

パイロットAB後データも開発に加える場合は、`pilot_post` のdatasetも指定し、`--include_pilot_post` を付けます。

```powershell
.\WPy64-312101\python\python.exe scripts\07_train.py `
  --data_roots datasets\normal_train,datasets\pilot_post_train `
  --label_source individual `
  --raters evaluator01 `
  --split_unit window `
  --mode skeleton `
  --include_pilot_post `
  --save_name models\normal_plus_pilot_post.pt
```

`fusion` を使う場合は、RGB特徴を弱めに混ぜます。

```powershell
.\WPy64-312101\python\python.exe scripts\07_train.py `
  --data_roots datasets\normal_train,datasets\pilot_post_train `
  --label_source individual `
  --raters evaluator01 `
  --split_unit window `
  --mode fusion `
  --fusion_image_scale 0.25 `
  --include_pilot_post `
  --save_name models\normal_plus_pilot_post_fusion.pt
```

`fusion` が `skeleton` より悪い場合は、`--fusion_image_scale 0.1` も試します。

## オフライン推論と注釈動画

`09_video_offline.py` で未知動画に推論します。

```powershell
.\WPy64-312101\python\python.exe scripts\09_video_offline.py `
  --ckpt models\normal_plus_pilot_post.pt `
  --mode skeleton `
  --data_root datasets\trial_post `
  --annotate_out outputs\trial_post_annotated.mp4 `
  --log_db outputs\pred_log.sqlite
```

`--annotate_out` で作成される注釈動画には、元動画の音声が付くようにしています。

処理の流れ:

1. OpenCVで無音の一時注釈動画を作成
2. ffmpegで注釈済み映像と元動画の音声を結合
3. 最終的な `annotated.mp4` として保存

元動画に音声がない場合やffmpeg結合に失敗した場合は、従来どおり無音の注釈動画が残ります。

## AB区間の後付け

ABの開始時刻・終了時刻が分かったら、`09_video_offline.py` の推論ログDBに後付けします。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\apply_ab_interval.py `
  --db outputs\pred_log.sqlite `
  --start 12:30 `
  --end 15:00
```

この処理は、元の `situation` を壊さず、以下の列を追加・更新します。

- `situation_override='AB'`
- `situation_override_note='Acti-Break'`

複数動画が同じDBに入っている場合は、`--video-filter` を使います。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\apply_ab_interval.py `
  --db outputs\pred_log.sqlite `
  --start 12:30 `
  --end 15:00 `
  --video-filter 20260714_unnan_nishi
```

## 集計

AB区間を後付けした後に、集計を再実行します。

```powershell
.\WPy64-312101\python\python.exe scripts\10_aggregate_predictions.py `
  --db outputs\pred_log.sqlite `
  --out_dir outputs `
  --bin_sec 5
```

`10_aggregate_predictions.py` は `situation_override` があれば、それを `situation` より優先して集計します。つまりAB区間は `AB` として集計されます。

出力:

- `agg_overall.csv`
- `agg_by_track.csv`
- `agg_timeseries_sec.csv`
- `agg_timeseries_bin.csv`
- `agg_timeseries_sec.xlsx`

## AB背景付きグラフ

集計後、背景塗りグラフを作成します。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\make_situation_bg_chart.py `
  --input outputs\agg_timeseries_sec.xlsx `
  --output outputs\agg_timeseries_sec_situation_bg.xlsx
```

`AB` は薄い赤系の背景で表示され、凡例にも表示されます。

## 配布用採点パッケージ

採点者配布用には以下を使います。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\build_labeling_package.py `
  --datasets datasets\lesson_001 datasets\lesson_002 `
  --output dist\concentration_labeler_package `
  --overwrite `
  --zip
```

`proxy_ids.mp4` が存在するdatasetでは、配布パッケージにも同梱されます。

## 現在の注意点

- 本試験の `pre` / `post` データは、学習には入れず推論・検証専用にする。
- `pilot_post` は、開発用に限って `--include_pilot_post` 付きで学習に入れる。
- `proxy_ids.mp4` を作り直したい場合は `build_proxy_id_video.py` を使う。
- AB区間を変更した場合は、`apply_ab_interval.py` と `10_aggregate_predictions.py` を再実行する。
- 背景グラフを更新したい場合は、最後に `make_situation_bg_chart.py` を再実行する。



