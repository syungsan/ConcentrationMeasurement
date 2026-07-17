# 集中度ラベリング・学習ワークフロー備忘録

このメモは、集中度ラベル付けアプリと学習パイプラインを、複数評価者・複数データセット運用に合わせて整備した内容のまとめです。

## 目的

授業動画から児童・生徒ごとの集中度を推定するために、教師または評価者が動画区間ごとに集中度ラベルを付け、そのデータを集約して機械学習モデルを訓練します。

主な運用想定は次の通りです。

- 代表者が先に授業状況を設定する
- 複数の評価者が同じデータセットを採点する
- 各評価者の `dataset.sqlite` を回収する
- 評価者ごとの採点結果をマージする
- 集中度ラベルの平均値を使って学習する
- 状況「聞く」「書く」「話し合う」も特徴量として使う

## 集中度スケール

集中度ラベルは 1〜7 の7段階です。

以前の 1〜10 スケールではなく、現在は以下のように扱います。

- 最小値: 1
- 最大値: 7
- GUIの入力ボタン: 1〜7
- DB制約: `score BETWEEN 1 AND 7`
- 予測値の整数化: 1〜7に丸め込み

動画上の `Class Avg` 表示は整数ではなく、小数第2位まで表示します。

例:

```text
Class Avg: 5.23   (n=12)
```

## 状況ラベル

授業状況は次の3種類に固定しています。

- 聞く
- 書く
- 話し合う

自由入力ではなく、トグル選択方式です。

状況ラベルは後段の学習特徴量として扱います。画像特徴量・姿勢特徴量に加えて、状況の one-hot または推定確率をモデルへ渡します。

## データベース構造の考え方

現在の推奨構造では、時間区間と評価者ラベルを分離しています。

主なテーブル:

| テーブル | 役割 |
|---|---|
| `videos` | 動画情報 |
| `windows` | 共有の時間区間と状況 |
| `segments` | 各window内の人物トラック |
| `segment_frames` | 各segmentに対応するフレーム・crop情報 |
| `labels` | 評価者ごとの集中度ラベル |
| `window_skips` | 評価者ごとのスキップ記録 |
| `label_events` | ラベル操作履歴 |
| `label_consensus` | マージ後に作られる平均ラベル |
| `merge_sources` | マージ元DBの記録 |

重要なのは、状況は評価者ごとではなく `windows` に保存する共有情報であることです。

## 代表者による状況設定モード

評価者が採点を始める前に、代表者が全区間の状況を設定します。

起動例:

```powershell
.\WPy64-312101\python\python.exe scripts\06_label_gui.py `
  --dataset datasets\lesson_001 `
  --task situation `
  --name representative
```

代表者モードでは、集中度の採点ではなく、時間区間ごとの状況を設定します。

全区間の状況を設定・確認した後、ロックします。ロックされていない状態では、評価者モードで採点できないようにしています。

## 評価者モード

評価者はロック済みの状況を見ながら、集中度だけを採点します。

起動例:

```powershell
.\WPy64-312101\python\python.exe scripts\06_label_gui.py `
  --dataset datasets\lesson_001 `
  --task rating `
  --name tanaka
```

現在の採点区間はデフォルト15秒です。

内部の基本windowが5秒単位の場合、評価画面では5秒×3個をまとめて15秒区間として表示・採点します。採点結果は内部の5秒segmentへ展開されます。

15秒にした理由は、30秒では区間内の集中度変動を見落とす可能性があるためです。

## ランチャー

プロジェクトルートに `launcher.py` があります。

起動用バッチ:

```powershell
launch_labeler.bat
```

または直接:

```powershell
.\WPy64-312101\python\python.exe launcher.py
```

ランチャーでは次のことができます。

- 管理者モード
  - 状況設定モードでGUIを起動
  - 評価者DBをマージ
  - 評価者用パッケージを作成
- 評価者モード
  - datasetsフォルダ内のdataset一覧から選択
  - 採点GUIを起動

評価者用配布パッケージでは、管理者メニューは非表示になります。

## 評価者用パッケージ

評価者に配布するためのパッケージは次のスクリプトで作成します。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\build_labeling_package.py `
  --datasets datasets\lesson_001 datasets\lesson_002 `
  --output dist\concentration_labeler_package `
  --overwrite
```

評価者用パッケージには、採点に必要なものだけを含めます。

- `launcher.py`
- `launch_labeler.bat`
- `scripts/06_label_gui.py`
- 必要な `scripts/lib`
- 対象dataset
- proxy動画
- 必要なcrop画像
- WinPythonランタイム
- `package_config.json`
- `RETURN_INSTRUCTIONS.txt`

評価者はパッケージ内のランチャーからdatasetを選び、採点します。

## 評価者DBの回収とマージ

評価者の採点が終わったら、各パッケージ内の次のファイルを回収します。

```text
datasets/<dataset名>/db/dataset.sqlite
```

回収時は、上書きを避けるため評価者名つきにリネームするのがおすすめです。

例:

```text
returned_dbs/
  lesson_001/
    tanaka_dataset.sqlite
    suzuki_dataset.sqlite
    sato_dataset.sqlite
```

マージ例:

```powershell
.\WPy64-312101\python\python.exe scripts\tools\merge_rater_databases.py `
  --base-db datasets\lesson_001\db\dataset.sqlite `
  --inputs returned_dbs\lesson_001\tanaka_dataset.sqlite returned_dbs\lesson_001\suzuki_dataset.sqlite `
  --output merged\lesson_001_merged.sqlite `
  --overwrite
```

マージ後には次が作られます。

```text
merged/lesson_001_merged.sqlite
merged/lesson_001_merged_consensus.csv
```

`label_consensus` には、評価者ごとのラベル平均、評価者数、標準偏差、最小値、最大値が保存されます。

同じPCで同じdatasetを複数人が採点する場合も、評価者名を分ければマージできます。

例:

```text
tanaka
suzuki
tanaka_2
```

同じ評価者名で同じsegmentに異なるスコアが入っている場合は、衝突としてマージを停止します。

## 学習

複数評価者の平均ラベルを使う場合は、`--label_source consensus` を指定します。

例:

```powershell
.\WPy64-312101\python\python.exe scripts\07_train.py `
  --data_roots datasets\lesson_001 `
  --db_paths merged\lesson_001_merged.sqlite `
  --label_source consensus `
  --mode fusion
```

個別評価者のラベルを使う場合は、`--label_source individual` と `--raters` を使います。

```powershell
.\WPy64-312101\python\python.exe scripts\07_train.py `
  --data_roots datasets\lesson_001 `
  --label_source individual `
  --raters tanaka,suzuki `
  --mode skeleton
```

## split_unit

`--split_unit` は、学習データと検証データをどの単位で分けるかを決める設定です。

| split_unit | 意味 | 用途 |
|---|---|---|
| `window` | 採点区間単位で分ける | 動作確認・開発初期 |
| `dataset` | dataset単位で分ける | 未知動画への汎化確認 |
| `session` | 授業・撮影セッション単位で分ける | 別授業への汎化確認 |
| `school` | 学校単位で分ける | 別学校への汎化確認 |

データが増えてきたら、最終評価は `session` または `school` を使うのが望ましいです。

例:

```powershell
--split_unit school
```

特定の学校を検証用に固定したい場合:

```powershell
--split_unit school --holdout_groups school:SCHOOL_B
```

## 状況背景つきグラフ

集中度推移のExcelに、状況ごとの背景色を付けるスクリプトがあります。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\make_situation_bg_chart.py
```

デフォルト入力:

```text
outputs/agg_timeseries_sec.xlsx
```

デフォルト出力:

```text
outputs/agg_timeseries_sec_situation_bg.xlsx
outputs/agg_timeseries_sec_situation_bg.png
```

ただし、このスクリプトには `openpyxl` と `Pillow` が必要です。

## トラブル対応メモ

### pip3.exe がブロックされる

`pip3.exe` がアプリケーション制御ポリシーで止まる場合は、直接 `pip3` を実行せず、Python経由でpipを呼びます。

```powershell
.\WPy64-312101\python\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Transformer学習中に NaN が出る

`07_train.py` では、評価時の `NaN` / `inf` 予測を検出し、QWK計算で落ちないようにしています。

ログに次が出る場合は、数値不安定です。

```text
val_nonfinite_pred=...
```

対策候補:

- learning rate を下げる
- batch size を下げる
- Transformerの層数やhidden sizeを小さくする
- 入力特徴量のNaN/infを確認する

### situation classifier の val_loss が NaN になる

ログ例:

```text
[situation:fusion] ep008 val_loss=nan val_acc=0.4103
```

これは分類器のlogitsにNaN/infが出ている可能性があります。現在は NaN batch を検出し、後段の回帰へNaNが伝播しにくいようにしています。

ログに次が出たら要注意です。

```text
val_nonfinite_logits=...
train_nonfinite_logits=...
train_skipped=...
```

## 実行時間ログ

主要スクリプトには、実行開始から終了までの所要時間表示を追加しています。

対象:

- `01_make_proxy.py`
- `02_detect_track.py`
- `03_make_segments.py`
- `04_extract_frames_assets.py`
- `06_label_gui.py`
- `07_train.py`
- `08_realtime.py`
- `09_video_offline.py`

表示例:

```text
所要時間: 01:15:47.07 (4547.07秒)
```

## ハードウェアメモ

機械学習の待ち時間短縮では、現在の構成が Ryzen 5 3400G + RTX 3050 6GB の場合、CPU交換よりGPU交換の方が効きやすい見込みです。

特に次の処理はGPUの影響が大きいです。

- 検出
- 姿勢推定
- 学習
- 動画推論

ただし、GPU交換時は電源容量と補助電源コネクタを確認する必要があります。

