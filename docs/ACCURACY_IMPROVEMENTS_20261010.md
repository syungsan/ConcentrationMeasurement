# 精度改善の実装と比較実験

追加した機能は改善候補であり、精度向上が確認済みという意味ではありません。従来の学習条件・旧チェックポイントとの比較のため、新しいオプションは原則として初期値を無効にしています。

## 追加オプション

| オプション | 処理 | 比較時の注意 |
|---|---|---|
| `--pose_features motion` | 肩中心・肩幅を基準にした関節位置、隣接フレームの変位、頭・胴・腕の骨方向、検出信頼度を入力 | COCO-17・uniform samplingが必要。眼球の視線、3D頭部角度、筆記の正解判定ではない |
| `--temporal_pool mean` | 最終フレームだけでなく、全時刻の時系列出力を平均 | 5秒・10フレームは維持 |
| `--ordered_ordinal` | 6つの閾値の確率に単調性を保証 | ordinalのみ。期待値を丸める従来の推論方法は維持 |
| `--score_loss_weight 0.2` | ordinal BCEに期待スコアのSmooth L1誤差を加える | 重みは要調整。regressionでは使用しない |
| `--class_balance` | 学習データのクラス頻度の逆平方根で損失を補正、最大3倍 | 少数クラスの改善とexact_accuracyは両立するとは限らない。検証ラベルで重みを作らない |
| `--backbone_lr_scale 0.1` | ImageNet事前学習済み画像backboneの学習率を他の層の1/10にする | pretrained特徴の急激な更新を抑制 |
| `--freeze_batchnorm` | BatchNormの移動平均・分散の更新を止める | affineパラメータは学習する。画像領域がImageNetから遠い場合は悪化し得る |
| `--lr_schedule plateau` | 検証指標が3epoch改善しない場合に学習率を半減 | 状況分類器は検証CE、集中度は選択指標に従う |
| `--early_stopping_patience 10` | 指定epoch数改善がなければ終了 | 各段階で最良モデルを保存 |
| `--selection_metric exact_accuracy` | 保存する最良モデルの基準を変更 | mse/mae/exact_accuracy/qwk。全比較で同じ基準にする |
| `--augmentation mild` | 区間全体で共通の小さな明度・コントラスト変換のみ | 左右反転、ランダム切り抜き、色相変更、ランダムグレー化を行わない |

骨格の変位はサンプリングされたフレーム間の相対移動です。物理単位の速度ではなく、欠落フレーム・検出誤差の影響も残ります。両肩や対象関節の信頼度が低いときは関連特徴をゼロにします。肩を基準に再正規化することで、フレームごとに見える関節が変わることによる中心・スケール変動の影響を減らします。

従来のraw骨格エンコーダを置換するオプションです。fusionでは画像と103次元の骨格特徴を融合し、imageモードには骨格特徴が追加されません。新設定はチェックポイントに保存され、DB評価・動画・リアルタイム推論で復元されます。

## 最初の学習例（PowerShell）

```powershell
python scripts/07_train.py `
  --data_roots datasets/20260227_unnan_nishi_5-1_1,datasets/20260227_unnan_nishi_5-1_2,datasets/20260311_unnan_nishi_5-1_1,datasets/20260313_unnan_nishi_5-1_1_a `
  --label_source individual --raters tadano --include_pilot_post `
  --mode fusion --temporal transformer --T 10 --sample_strategy uniform `
  --image_roi upper_body --fusion_image_scale 0.5 `
  --pose_features motion --temporal_pool mean `
  --ordered_ordinal --score_loss_weight 0.2 `
  --lr 1e-4 --backbone_lr_scale 0.1 --freeze_batchnorm `
  --lr_schedule plateau --epochs 50 --early_stopping_patience 10 `
  --augmentation none --situation_epochs 10 --situation_teacher_forcing 0 `
  --situation_profile legacy `
  --split_unit dataset --holdout_groups dataset:0 `
  --selection_metric mse `
  --save_name models/improved_fusion_motion.pt
```

この例では最初のデータセットを丸ごと検証に取り、他の3つで学習します。従来の「4つ全部からwindow単位で分割したモデルを、学習元の1つで評価する」数値とは直接比較できません。データセット間で同じ児童が映る可能性はあり、未見児童・他校への汎化を保証する分割でもありません。

検証データは最良epochを選ぶために使用します。実用精度の最終報告には、モデル・設定選択に使用しない別の授業・学校を確保してください。

## 条件を切り分ける比較実験

```powershell
python scripts/tools/run_accuracy_experiments.py `
  --data_roots datasets/20260227_unnan_nishi_5-1_1,datasets/20260227_unnan_nishi_5-1_2,datasets/20260311_unnan_nishi_5-1_1,datasets/20260313_unnan_nishi_5-1_1_a `
  --raters tadano --variants baseline,optimization,motion,ordered,balanced,mild,no_situation,grayscale `
  --seeds 42 --output_dir outputs/accuracy_comparison_20261010 `
  --execute
```

`--execute`を省くと、コマンドの表示と実験計画JSONの保存だけを行います。すべて同じholdout・seed・最大epoch・選択基準で比較します。baselineは新しい改善を無効にした対照で、augmentation none・teacher forcing 0です。optimizationは学習調整の組、motionは骨格特徴、orderedは順序保証とスコア損失を加えます。balanced/mild/no_situation/grayscaleはorderedから各要素を変えます。旧実験の完全再現ではありません。

各実験のログ・独立チェックポイント・学習履歴に加え、最良MSEのepochの各指標を`summary.csv`に保存します。既存チェックポイントがある出力先には上書きしません。まずseed42で候補を絞り、良い条件を`--seeds 42,43,44`で再確認してください。自動実験はfusionモデルを8条件で学習するため、GPUでも時間がかかります。クラス補正やAugmentationを常に良いものとして採用しないでください。

検証exact_accuracy・±1一致率・MAE・QWK・学習クラス分布・学習多数クラスを常に予測した検証baselineを記録します。多数クラスは学習ラベルのみから決定します。

## 今回の範囲と次の改善

既存の保存済み画像・骨格から実装できる改善を加えました。顔ランドマークによる頭部姿勢・高解像度の眼球視線・手指や筆記認識は、現在のCOCO-17データから正確に復元できないため未実装です。元映像の顔解像度・撮影角度・遮蔽を調べてから専用検出器を導入する必要があります。

長時間文脈を加える場合も、児童追跡の連続性とラベルの時間的対応を確認する必要があります。今回、5秒区間を無条件につなげる処理は加えていません。複数評価者の一致度、授業状況ごとの誤り、児童・授業間のラベル分布差も、モデル変更とは別に確認すべき項目です。

## 動作検証

既存の監査テスト11件と、unittestで実行する32件（追加8件を含む）が通りました。旧・新チェックポイントの推論一致、単調な閾値、特徴量の並進・スケール不変性、低信頼度マスク、勾配、弱いAugmentation、比較実験の出力と上書き防止、学習・状況分類の結合を検証しています。

実データでも固定seedで抽出した学習64区間・検証32区間を使い、骨格motionモードで各1epochの状況分類・集中度学習・保存を確認しました。記録は`outputs/improvement_real_smoke.json`と`outputs/improvement_real_smoke_split.json`です。この少量・短時間の実行は精度比較の根拠にしません。全件の1epoch確認は読み込みに時間がかかったため中止し、少量の動作確認へ切り替えています。8条件の本学習は未実施です。

## 分類器の悪化報告を受けた修正

比較スクリプトは学習出力を端末とログファイルへ同時に表示します。実験番号・完了件数・epoch・25batchごとのbatch番号・経過時間・検証開始が分かります。出力が15秒途切れた場合にも実行中の通知を表示します。全体の正確な残り時間は推定していません。変更は次回起動から有効で、既に実行中のプロセスには適用されません。

`--situation_profile legacy`で、シチュエーション分類器の骨格入力をraw、時系列集約をlast、backbone学習率倍率を1、BatchNorm凍結なし、学習率schedulerなし、早期終了なし、Augmentationなしに分離します。集中度モデルの変更は維持できます。画像ROI・グレースケール・T・temporal方式は集中度モデルと共有します。このため、任意の古いモデルを完全再現する設定という意味ではありません。`07_train.py`では互換性のためsharedが初期値ですが、比較スクリプトはlegacyが初期値です。新しい分類器設定はチェックポイントに保存して推論時に復元し、設定がない以前のモデルは従来の読み方を維持します。

確認した`models/improved_fusion_motion_202610102002.pt`はdataset:0 holdoutで、保存された分類器の検証精度は55.79%、最大検証精度は57.88%でした（保存基準は最小CE）。集中度は最初のepochでexact 38.44%、MSE 2.878、その後学習損失は0.562から0.082まで減少しましたが、検証MSEは最終epochで3.578でした。過学習や授業間の差が疑われます。

一方、改善オプションなしの比較baselineも分類器精度は最初のepochで45.55%、以降も低下して途中終了しています。したがって追加特徴だけが原因と断定できません。以前のwindow分割とは異なる難しい評価です。従来条件で変更の効果を比較するには、比較スクリプトに`--split_unit window`を指定するとseedによるwindow分割になります（holdout_groupsは指定しない）。この分割の高精度は未見授業での精度を意味しません。

従来の`evaluate_labeled_dataset.py`は引き続き使用できます。新モデル・以前のv1〜v5チェックポイントを既存の共通loaderで読みます。改修で既存ファイルの学習結果が改善するわけではなく、legacy分類器での再学習が必要です。

```powershell
python scripts/tools/evaluate_labeled_dataset.py `
  --model models/improved_fusion_motion_202610102002.pt `
  --data_roots datasets/20260227_unnan_nishi_5-1_1 `
  --raters tadano --situation-source predicted `
  --output-dir outputs/evaluation_improved_auto
```

`--situation-source true`に変えれば、DBの正解シチュエーションで評価できます。autoとの比較で分類器の影響を調べられます。状況特徴を無効化したモデルではこの比較は集中度予測に影響しません。
