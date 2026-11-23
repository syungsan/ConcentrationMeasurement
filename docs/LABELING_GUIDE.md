# ラベリングとデータセット運用

## 目的と基本仕様

授業動画を時間区間（Window）へ分割し、代表者が授業状況を設定した後、複数の評価者が児童ごとの集中度を採点します。

- 集中度: 1（低い）～7（高い）の7段階
- 状況: `聞く`、`書く`、`話し合う`
- 基本Window: 通常5秒
- 採点画面: 連続する基本Windowを既定15秒にまとめて表示
- 採点結果: 表示区間内の基本segmentへ展開して保存

状況は評価者固有ではなく`windows`の共有情報です。集中度は`labels`へ評価者名とともに保存されます。

## データ準備

番号付きスクリプトを順に実行します。

| 順序 | スクリプト | 役割 |
|---:|---|---|
| 1 | `scripts/01_make_proxy.py` | 元動画からproxy動画を作成 |
| 2 | `scripts/02_detect_track.py` | 人物検出・追跡、ID付きproxyを作成 |
| 3 | `scripts/03_make_segments.py` | Windowと人物segmentを作成 |
| 4 | `scripts/04_extract_frames_assets.py` | cropと姿勢データを作成 |
| 5 | `scripts/05_configure_research_dataset.py` | 必要に応じて研究メタデータを設定 |

`videos/proxy_ids.mp4`が存在する場合、ラベラーは`proxy.mp4`より優先して再生します。既存の検出結果から作り直す例:

```powershell
.\WPy64-312101\python\python.exe scripts\tools\build_proxy_id_video.py `
  --dataset datasets\lesson_001 `
  --label-fps 4
```

## ランチャー

通常はプロジェクト直下のBATから起動します。

```powershell
.\launch_labeler.bat
```

直接起動する場合:

```powershell
.\WPy64-312101\python\python.exe launcher.py
```

## 状況設定

代表者が全Windowの状況を設定し、確認後にロックします。CLI起動例:

```powershell
.\WPy64-312101\python\python.exe scripts\06_label_gui.py `
  --dataset datasets\lesson_001 `
  --task situation `
  --name representative
```

ロックされていないWindowは評価者が採点できません。状況境界をまたいで採点区間が結合されることはありません。

## 集中度採点

```powershell
.\WPy64-312101\python\python.exe scripts\06_label_gui.py `
  --dataset datasets\lesson_001 `
  --task rating `
  --name evaluator01
```

主な操作:

- 数字キー`1`～`7`: 集中度を入力
- `N`: 次の未完了区間
- `PageUp` / `PageDown`: 前後の区間
- シークバー: 離した時刻に対応する編集区間へ移動
- スキップ: 採点対象外として記録（後から解除可能）
- 集計を見る: 評価者自身の入力状況を確認

「同じ状況を優先」を有効にすると、直近ではなく同じ状況の未完了区間へ移動する場合があります。

## 評価者用パッケージ

代表者が状況を確定・ロックした後に作成します。ランチャーから複数datasetを選択して作成するか、CLIを使用します。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\build_labeling_package.py `
  --datasets datasets\lesson_001 datasets\lesson_002 `
  --output dist\concentration_labeler `
  --overwrite `
  --zip
```

評価者名は起動時に空欄で表示され、評価者が入力します。採点後は各パッケージの次のDBを回収します。

```text
datasets/<dataset名>/db/dataset.sqlite
```

上書きを避けるため、回収時は評価者名を付けて保管してください。

```text
returned_dbs/lesson_001/evaluator01_dataset.sqlite
returned_dbs/lesson_001/evaluator02_dataset.sqlite
```

## 評価者DBのマージ

ランチャーの管理者メニュー、または次のコマンドでマージします。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\merge_rater_databases.py `
  --base-db datasets\lesson_001\db\dataset.sqlite `
  --inputs returned_dbs\lesson_001\evaluator01_dataset.sqlite returned_dbs\lesson_001\evaluator02_dataset.sqlite `
  --output merged\lesson_001_merged.sqlite `
  --overwrite
```

マージ後のDBには評価者別の`labels`と合意ラベル`label_consensus`が保存されます。同じ評価者・同じsegmentに矛盾する値がある場合や、異なるdatasetのDBを混ぜた場合はエラーになります。

## マージ済みDBから評価者を削除する

評価者の名前と、評価者に紐づくテーブルごとの件数を確認します。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\remove_rater.py list --db merged\lesson_001_merged.sqlite
```

一覧で確認した名前を `--rater` に指定します。複数人を削除する場合は名前をスペース区切りで並べ、空白を含む名前は引用符で囲みます。名前は大文字・小文字や空白も含めて完全一致で指定します。見つからない名前が一つでもある場合は、処理全体が中止されます。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\remove_rater.py remove `
  --db merged\lesson_001_merged.sqlite --rater "evaluator01" "evaluator02" `
  --output merged\lesson_001_cleaned.sqlite
```

指定した評価者の採点、観察情報、操作履歴、スキップ、ブラインド割り当て、除外指定を削除したDBを別ファイルに保存し、評価者ごとの削除件数を表示します。残りの採点から合意ラベルを再計算し、同じフォルダーに `lesson_001_cleaned_consensus.csv` を出力します。出力先のDBまたはCSVがすでにある場合はエラーになります。共有状況と `merge_sources` の取り込み履歴は保持します。

## 主なDBテーブル

| テーブル | 内容 |
|---|---|
| `windows` | 時間区間と共有状況 |
| `segments` | Window内の人物トラック |
| `segment_frames` | segmentに対応するcrop・フレーム情報 |
| `labels` | 評価者別の集中度 |
| `window_skips` | 評価者別のスキップ記録 |
| `label_events` | 採点操作履歴 |
| `label_consensus` | マージ後の合意ラベル |
| `label_observations` | 採点時の表示区間などの観察情報 |
