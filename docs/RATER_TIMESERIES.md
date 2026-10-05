# 評価者別の集中度時系列グラフ

06_label_gui.py で評価した dataset の SQLite DB から、評価者の名前付き折れ線グラフを作成します。単独評価と複数人評価の両方に対応します。

## 起動方法

プロジェクトのルートから実行します。引数なしでは DB の選択画面が開き、選択後にグラフを保存して表示します。

```powershell
python scripts/tools/plot_rater_timeseries.py
```

dataset を指定する場合:

```powershell
python scripts/tools/plot_rater_timeseries.py --dataset_root datasets/lesson_001 --show
```

DB を直接指定する場合（複数評価者をまとめた DB も使用できます）:

```powershell
python scripts/tools/plot_rater_timeseries.py --db merged/lesson_001_merged.sqlite --show
```

特定の評価者・児童だけ表示する場合:

```powershell
python scripts/tools/plot_rater_timeseries.py --dataset_root datasets/lesson_001 --raters teacherA teacherB --track_id 10 --show
```

評価者名に空白がある場合は引用符で囲みます。--show を省略すると保存のみ行います。--output_dir outputs/lesson_001_ratings で保存先、--title "授業1の集中度" でタイトルを指定できます。

## グラフの読み方

- 横軸: 動画内の経過時間（分）。各評価区間の中央時刻に点を置きます。評価を入力した日時ではありません。
- 縦軸: 集中度（1～7）。
- 凡例: DB に記録された各評価者の名前。評価者ごとに色を変えます。
- 既定の集計: 各時間区間で、その評価者が評価した対象児童の点数を単純平均します。
- --track_id 指定時: 指定した児童の点数をそのまま表示します。track_id は動画単位の追跡 ID です。

未評価・スキップ区間は欠測として線を途切れさせます。対象外人物（excluded_segments）の評価は除外します。DB の時間区間自体に隙間がある場合も線を接続しません。動画が複数ある DB では、動画ごとに別のグラフを生成します。

平均の対象児童数は、評価者や時間区間によって異なる可能性があります。評価者間の差には、採点の違いだけでなく対象児童の違いも含まれるため、CSV の n_labeled も確認してください。同じ児童の評価を比較する場合は --track_id を指定します。

## 保存ファイル

既定の保存先は、入力 DB の隣にある ratings_timeseries フォルダです。

- concentration_video_<video_id>.png: 画像として利用する折れ線グラフ。
- concentration_video_<video_id>.svg: 拡大・編集に適したベクター形式のグラフ。
- ratings_timeseries.csv: video_id、window_id、区間開始・終了秒、評価者名、集中度、対象人数 n_labeled。欠測の score は空欄になります。

同じ保存先で再実行すると、同名の出力ファイルを更新します。入力 DB は読み取り専用で開き、評価データを変更しません。

## 評価者ごとに DB が分かれている場合

このツールは1つの DB 内の評価者を比較します。配布・回収した DB が評価者ごとに別々の場合は、既存の scripts/tools/merge_rater_databases.py で同じ dataset の DB をまとめ、その出力を --db で指定してください。

必要なライブラリは既存の matplotlib と、DB 選択画面に使う PySide6 です。GUI を使わず保存だけ行う場合は PySide6 を読み込みません。

## 平滑化オプション

折れ線の細かな上下動を抑えたい場合は、--smooth_window 3 を指定します。

```powershell
python scripts/tools/plot_rater_timeseries.py --dataset_root datasets/lesson_001 --smooth_window 3 --show
```

DB を選択画面から選ぶ場合も指定できます。

```powershell
python scripts/tools/plot_rater_timeseries.py --smooth_window 3
```

- 1（既定値）: 平滑化なし。
- 3: 前後1区間と現在の区間、計3区間の中心移動平均。軽い平滑化。
- 5: 前後2区間と現在の区間、計5区間の中心移動平均。より強い平滑化。

1以上の奇数を指定します。各区間を同じ重みで平均し、評価者・動画ごとに独立して処理します。区間長が5秒なら3区間は通常15秒分、5区間は25秒分です。秒数ではなく区間数の指定なので、区間長が異なる場合は時間幅も異なります。

未評価・スキップ区間や時間の隙間をまたいで平均しません。連続する評価区間の端では、その範囲内にある点だけで平均します。欠測点は欠測のまま残します。

平滑化は表示・保存する PNG/SVG の折れ線にだけ適用し、CSV は元の集計値を保存します。グラフのタイトルに移動平均の区間数を記載します。平滑化によって短時間のピークは小さくなるため、採点値の確認には CSV または --smooth_window 1 を使ってください。同じ保存先で実行すると既存のグラフを更新します。
