# 研究設計・集計・AB解析

## 研究上の対象

本プロジェクトは、映像から観察できる「授業課題への従事度」を集中度として扱います。内面的な集中そのものを直接測定するものではありません。モデル開発用データと、介入効果を評価する本試験データは分離します。

## データセットの研究区分

```powershell
.\WPy64-312101\python\python.exe scripts\05_configure_research_dataset.py `
  datasets\school_a_grade5 `
  school_a `
  5 `
  5A `
  normal
```

| phase | 意味 | 学習への使用 |
|---|---|---|
| `normal` | 通常の開発用授業 | 使用可能 |
| `pilot_post` | 独立したパイロット介入後 | `--include_pilot_post`指定時のみ |
| `pre` | 本試験の介入前 | 使用禁止 |
| `post` | 本試験の介入後 | 使用禁止 |

本試験開始後にモデル、しきい値、除外基準を調整しないでください。未知学校で評価する場合は学校単位で分割し、可能なら3校以上の開発データを用意します。

## 盲検採点

評価者用パッケージには介入条件、撮影順序、期待される効果を含めません。評価者は総合的な課題従事度のみを1～7で入力します。最低2名、重要な外部検証では3名の独立評価を推奨します。

採点とDB回収の詳細は[LABELING_GUIDE.md](LABELING_GUIDE.md)を参照してください。

## 教室全体の集計

推論ログを時間単位で集計します。

```powershell
.\WPy64-312101\python\python.exe scripts\10_aggregate_predictions.py `
  --db outputs\pred_log.sqlite `
  --out_dir outputs\lesson01 `
  --bin_sec 5
```

各時点で児童内平均を求めてから教室全体を集約します。外れ値に強い`score_trimmed_mean`を主指標候補とし、平均、中央値、標準偏差、検出人数も併記します。人手評価と比較する場合は、同じ児童・同じ時間範囲を揃えてください。

## AB区間の登録

アクティ・ブレイク（AB）の開始・終了を推論DBへ後付けできます。

```powershell
.\WPy64-312101\python\python.exe scripts\tools\apply_ab_interval.py `
  --db outputs\pred_log.sqlite `
  --start 12:30 `
  --end 15:00
```

複数動画が同じDBにある場合は`--video-filter`を指定します。元の授業状況は保持され、overrideとしてABが記録されます。変更後は集計を再実行してください。

## AB前後レポート

```powershell
.\WPy64-312101\python\python.exe scripts\13_ab_intervention_report.py `
  --db outputs\pred_log.sqlite `
  --output outputs\ab_intervention_report.xlsx
```

既定の解析条件:

- AB前300秒とAB後300秒を比較
- AB実施中は主要効果量から除外
- AB終了後90秒をクールダウンとして除外
- 5秒単位で児童内平均後、教室全体を集約
- 状況別比較と、AB前の状況構成比による標準化を実施
- クールダウン0、30、60、90、120、180秒で感度分析

ABイベントが1件だけの場合、結果は設定感度を確認する予備解析です。因果効果の確定には、事前登録した設計、複数イベント、クラスターを考慮した推定が必要です。

## 介入効果の予備解析

クラス・時点単位のCSVを入力します。

```powershell
.\WPy64-312101\python\python.exe scripts\11_intervention_analysis.py `
  --input outputs\trial_classroom_timeseries.csv `
  --out-dir outputs\trial_analysis
```

単純な前後差だけでなく、可能なら非介入群を用いた差の差を確認します。

```text
介入効果 = 介入群の前後差 - 非介入群の前後差
```

探索的解析と確認的解析を区別し、確認的解析ではクラスター無作為化、ステップウェッジ、または学校・学年・クラスを考慮した混合効果モデルを使用します。

## 測定不変性の確認

介入前後でモデル誤差が系統的に変化していないか監査します。入力CSVには`school_id,phase,human_score,model_score`が必要です。

```powershell
.\WPy64-312101\python\python.exe scripts\12_measurement_invariance_audit.py `
  --input outputs\pilot_human_model.csv `
  --output outputs\pilot_measurement_audit.json
```

`post_minus_pre_bias`の信頼区間が大きく0から外れる場合、介入後だけモデルが高くまたは低く判定している可能性があります。独立したパイロットデータで再学習・再監査してください。

## 時系列比較の注意

- 人手評価が瞬間値か、直前区間か、直後区間かを事前に定義する。
- 評価用の局所平均と、表示用の移動平均を混同しない。
- 長い300秒平均は短時間の変化を薄めるため、10～300秒の複数窓で感度分析する。
- 最終端の短い区間を通常区間と同等に扱わない。
- 教室全体平均と一部児童の人手評価を直接比較しない。

## データ管理

児童映像は同意、保存期間、アクセス権、匿名化方針を研究開始前に確定します。介入の実施中と直後の残留運動を測る区間も事前定義し、実施時刻を記録してください。
