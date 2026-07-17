# 教室集中度・介入効果研究ワークフロー

このプロジェクトが測定する対象は、児童の内面的な集中そのものではなく、映像から観察可能な「授業課題への従事度」です。モデル開発用データと介入効果を検証する本試験データを分離し、本試験開始後はモデルを固定してください。

## 実行順

1. `01_make_proxy.py`
2. `02_detect_track.py`
3. `03_make_segments.py`
4. `04_extract_frames_assets.py`
5. `05_configure_research_dataset.py`
6. `06_label_gui.py`
7. `07_train.py`
8. `08_realtime.py`
9. `09_video_offline.py`
10. `10_aggregate_predictions.py`
11. `11_intervention_analysis.py`
12. `12_measurement_invariance_audit.py`

`05`の必須入力はデータセット、匿名学校ID、学年、匿名学級ID、区分の5つです。`pilot_post`と`post`のみ、介入後の経過分を追加します。

## 1. データ区分

各 `dataset.sqlite` に学校・学級・セッション・介入区分を登録します。

```powershell
WPy64-312101\python\python.exe scripts\05_configure_research_dataset.py `
  datasets\school_a_grade5 school_a 5 5A normal
```

区分は次のとおりです。

| phase | 意味 | 学習への使用 |
|---|---|---|
| `normal` | 通常授業 | 使用可 |
| `pilot_post` | 独立したパイロットの療法後 | `--include_pilot_post` 指定時のみ使用可 |
| `pre` | 本試験の介入前 | 使用禁止 |
| `post` | 本試験の介入後 | 使用禁止 |

`pre`と`post`は自動的に本試験用として登録され、学習から除外されます。`pilot_post`と`post`では`--minutes-after`を指定します。

## 2. 盲検採点

配布パッケージ作成時に評価者を決める必要はありません。配布先で評価者名を初めて入力した時点で、その評価者専用のランダム順が自動生成されます。

GUIは次のように起動します。

```powershell
WPy64-312101\python\python.exe scripts\06_label_gui.py `
  --dataset datasets\school_a_grade5 --task rating --name evaluator01 --blind-order
```

採点GUIは既定で、連続する5秒区間を最大30秒にまとめて再生します。ただし「聞く・書く・話し合う」の境界はまたぎません。児童に1回付けた点数は、その30秒内に存在する同一児童の5秒segmentへ展開されるため、学習・解析は従来どおり5秒粒度です。元の5秒採点へ戻す場合は `--rating-clip-sec 5`、20秒に変更する場合は `--rating-clip-sec 20` を指定します。

30秒評価から展開されたラベルには、元クリップの開始・終了時刻も `label_observations` に保存されます。統計解析では、同じクリップ由来の6件を完全に独立した6評価とは扱わないでください。

配布パッケージでは盲検順を自動生成し、介入前後のメタデータを含めません。評価者には介入条件、撮影順、期待される効果を伝えないでください。

採点時には1～7の総合課題従事度だけを入力します。入力負担を下げるため、観察不能フラグや確信度は扱いません。原則として各映像を2名以上、重要な外部検証映像は3名で採点してください。

RGB画像を使う場合も、全身画像を強く使うと服装・体格・教室背景に引っ張られやすいため、保存cropは採点確認用に全身のまま残し、モデル入力時だけ頭〜上半身ROIに切ります。`fusion` モデルではRGB特徴を補助情報として弱く混ぜ、skeleton特徴を主入力として扱います。

## 3. 学習

学校単位の検証を既定としています。最低2校、望ましくは3校以上の開発データが必要です。

```powershell
python scripts\07_train.py `
  --data_roots datasets\school_a,datasets\school_b,datasets\school_c `
  --db_paths merged\a.sqlite,merged\b.sqlite,merged\c.sqlite `
  --label_source consensus --min_raters 2 --max_label_std 1.5 `
  --split_unit school --objective ordinal --mode skeleton `
  --save_name models\school_generalization.pt
```

この学習では集中度モデルに加え、管理者が設定した `聞く・書く・話し合う` を教師データとするシチュエーション分類器も同じチェックポイントへ保存されます。既定では分類器を10エポック学習し、集中度モデルの半分のバッチには分類器が推定した確率を入力します。

## 4. 自動シチュエーション推定

新しく学習したチェックポイントでは、`--situation` を省略すると自動推定になります。

```powershell
python scripts\08_realtime.py `
  --ckpt models\school_generalization.pt --mode skeleton --source 0 --show

python scripts\09_video_offline.py `
  --ckpt models\school_generalization.pt --mode skeleton `
  --data_root datasets\unseen_lesson --annotate_out outputs\annotated.mp4
```

各児童の時系列から得た3分類確率を教室全体で平滑化し、その確率を集中度モデルへ入力します。ログには採用分類、最大確率、3分類確率を保存します。`--situation 聞く` のように指定すると手動上書きできます。旧チェックポイントには分類器がないため、自動推定には再学習が必要です。

パイロット療法後データを開発に加えるときだけ `--include_pilot_post` を指定します。本試験データは指定しても除外されます。

比較すべきモデルは、`skeleton`、`image`、`fusion` です。未知校でRGBモデルだけ性能低下する場合は、学校固有の服装・背景を利用している可能性があるため、姿勢モデルを主解析にします。

評価指標はRMSE、MAE、R²に加え、Spearman相関と二次重み付きκを出力します。モデル選択と除外基準は本試験前に固定してください。

## 5. 教室全体の集約

```powershell
python scripts\10_aggregate_predictions.py `
  --db outputs\pred_log.sqlite --out_dir outputs\lesson01 --bin_sec 30
```

各時間区間について、児童内で平均してから教室全体を集約します。出力には平均、10%トリム平均、中央値、標準偏差、低得点（3以下）割合、検出児童数が含まれます。主指標には外れ値に強い `score_trimmed_mean` を推奨します。

## 6. 療法前後で測定誤差が変わらないか確認

パイロットデータについて、盲検された人間の合意点と固定モデルの予測を1行ずつ並べたCSVを作ります。必須列は `school_id,phase,human_score,model_score` です。

```powershell
python scripts\12_measurement_invariance_audit.py `
  --input outputs\pilot_human_model.csv `
  --output outputs\pilot_measurement_audit.json
```

`post_minus_pre_bias` の信頼区間が大きく0から外れる場合、療法後だけモデルが系統的に高く／低く判定しています。その場合は独立した `pilot_post` を追加して再学習し、別のパイロット校で再監査します。

## 7. 介入効果の予備解析

固定モデルによる教室集約を、必須列 `school_id,class_id,session_id,phase,time_bin,classroom_score` のCSVにまとめます。

```powershell
python scripts\11_intervention_analysis.py `
  --input outputs\trial_classroom_timeseries.csv `
  --out-dir outputs\trial_analysis
```

このスクリプトは学級を単位とした対応あり差と、学級クラスタ・ブートストラップ信頼区間を出します。これは予備解析です。確認的な結論には、事前登録したクラスター無作為化またはステップウェッジ設計と、暦時・教科・学年・学校／学級を考慮した混合効果モデルを使用してください。

## 運用上の必須事項

- 療法の実施中と、療法直後の残留動作を測る区間を分ける。
- 例: 0～2分、2～10分、10～20分を事前に定義する。
- 本試験データでモデルや閾値を再調整しない。
- 未知校では少量の盲検採点を行い、外部妥当性を確認する。
- 映像指標だけで結論を出さず、課題成績や盲検観察など独立指標を併用する。
- 児童映像・顔画像・識別子について、同意、保存期間、アクセス権、匿名化方針を研究開始前に確定する。

## 推奨する撮影時間

- モデル開発用の通常授業は、可能なら授業全体（小学校では約40～45分）を撮影する。
- 1本を長くすることより、教科・教師・曜日・学年が異なる複数授業を集めることを優先する。
- パイロット療法後と本試験後は、持続効果を見るため最低20分、可能なら30分程度を確保する。
- 介入直後0～2分、2～10分、10～20分を区別できるよう、介入終了時刻を記録する。
- 10分未満の動画だけでは、一時的な覚醒と持続的な授業集中を分けにくい。
