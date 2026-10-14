# 条件選択、クロスバリデーション、全データ学習

## 時間を抑えた比較

初めから全条件・全seedをCVする必要はありません。window分割で良かった2条件を、1seed・2foldで比較できます。

```powershell
python scripts/tools/run_accuracy_experiments.py `
  --data_roots $trainingRoots --raters tadano `
  --cross_validation dataset --cv_folds 2 `
  --variants no_situation,mild --seeds 42 `
  --selection_metric exact_accuracy --epochs 50 --situation_epochs 20 `
  --concentration_patience 5 --situation_patience 3 `
  --output_dir outputs/cv_quick --execute
```

4データセット・3条件・3seedの36回から、2fold・2条件・1seedの4回へ減らせます。さらに集中度は5epoch、分類器は3epoch、検証指標が更新されなければ早期終了します。最良モデルを保存します。上限の50/20を必ず実行するわけではありません。早期終了を短くしすぎると後から改善するepochを見逃す可能性があります。

`--cv_folds 2`は入力順で0・2・4…番目と1・3・5…番目のデータセットに分けます。同じデータセットを学習・検証の両方には配置せず、各データセットを一度ずつ検証します。データセット件数を均等に振り分け、区間数・状況の割合は均等とは限りません。`--cv_folds 3`なら3fold、0または省略なら従来のN回です。

2foldでは各回の学習対象が約半分になり、N−1データセットで学習する以前の検証より少なくなります。精度や推奨epochは以前の分割と直接比較しません。まず大きな差があるか確認し、僅差なら残した条件だけ追加seedや3foldで確認してください。学習損失やデータ量を減らして高精度に見せる変更ではなく、実験予算を抑えるための設定です。

データセット数を4に固定しません。`--cross_validation dataset`は、入力されたN個のデータセットから1個ずつを検証に取り、残りN−1個で学習するN回の比較を実施します。同じ児童が異なるデータセットに映る場合は、未見児童での検証ではありません。Nが大きくなると実行回数も増えるため、まずwindow分割で候補を絞ります。

## 1. window分割で条件を比較

PowerShellで対象を変数に入れます。データセットが増えたら、このカンマ区切りの一覧を更新してください。

```powershell
$trainingRoots = "datasets/20260227_unnan_nishi_5-1_1,datasets/20260227_unnan_nishi_5-1_2,datasets/20260311_unnan_nishi_5-1_1,datasets/20260313_unnan_nishi_5-1_1_a"

python scripts/tools/run_accuracy_experiments.py `
  --data_roots $trainingRoots --raters tadano `
  --split_unit window --variants baseline,optimization,motion,ordered,balanced,mild,no_situation,grayscale `
  --selection_metric exact_accuracy --epochs 50 --situation_epochs 20 `
  --output_dir outputs/window_screening --execute
```

window分割は既定で80%学習・20%検証です。授業全体を検証用に取り置きません。同じ時刻に映る児童はまとめて片側に配置します。ただし同じ児童・隣接時間が両側に入り、高い精度が出ても未見授業への汎化を保証しません。

50・20は上限であり、最良epochを固定する値ではありません。集中度は指定した選択指標、シチュエーション分類器は最小検証CEのepochを採用します。比較スクリプトの分類器はlegacy設定で、集中度向けの骨格・最適化変更を適用しません。legacy分類器には早期終了を適用せず、20epochを比較します。すべての条件で同じseed・分割・上限・選択指標にそろえます。

`summary.csv`を見てexact_accuracyが高い候補を2〜3個残します。MAE、QWK、分類器の精度も確認してください。僅差なら改善前のbaselineを残して比較します。学習損失の小ささで条件を選びません。最良epochが上限に達した場合は、上限が足りない可能性があり、その条件について上限を増やして再確認します。

## 2. N個の授業を交代して検証

以下のbaselineとmotionは例です。第1段階で残した条件に置き換えます。

```powershell
python scripts/tools/run_accuracy_experiments.py `
  --data_roots $trainingRoots --raters tadano `
  --cross_validation dataset --variants baseline,motion `
  --selection_metric exact_accuracy --epochs 50 --situation_epochs 20 `
  --seeds 42,43,44 --output_dir outputs/lesson_cv --execute
```

Nデータセット×条件数×seed数の学習を行います。N=6、2条件、3seedなら36回です。最初は`--seeds 42`で実行して候補を絞ることもできます。各foldのチェックポイント・ログ・最良epoch・指標を保存します。

`cv_recommendation.json`には次を出力します。

- 各条件の授業単位exact_accuracyの平均と標準偏差、MAE・QWKの平均
- 推奨条件：平均exact_accuracy最大、同点なら平均MAE最小
- 集中度の推奨epoch数：その条件の各fold・seedの最良epochの中央値を切り上げ
- 分類器の推奨epoch数：各fold・seedの最小検証CEのepochの中央値を切り上げ

たとえば集中度の最良epochが3、5、20なら、推奨は5です。中央値は実用上の目安で、全データでの最適epochを保証しません。授業によって精度や最良epochが大きく変わる場合は、一律のepoch数よりデータ・ラベル・授業状況の違いを先に調べます。条件選択にも使ったCV平均は、独立した最終テスト精度ではありません。

## 3. 選んだ条件とepochで全データを学習

CV終了時に`final_training_command.json`へ、選んだ条件と推奨epochを使うコマンドを保存・表示します。そのコマンドには`--no_validation`が付いています。

CV終了後の全データ学習まで自動実行する場合は、第2段階のコマンドに`--final_train`を追加します。通常実行では、選択した条件・seed・foldの既存チェックポイントとログを上書きします。

全データ学習だけを直接実行するなら、通常の`07_train.py`コマンドに次を追加します。5・3は例なのでCVの推奨値に置き換えます。

```powershell
--no_validation --epochs 5 --situation_epochs 3
```

集中度・分類器の両方で検証を省き、採用可能な全区間を学習します。除外区間、評価者、研究データの採用条件は引き続き適用されます。holdout指定は無視します。検証に依存する早期終了とplateau schedulerは無効にし、指定epochの最終状態を保存します。検証MSEなどは計算せず、チェックポイントの集中度resultsはnullになります。

CVでplateauによる学習率低下が有効だった条件を全データで再学習する場合、その学習率の経過は再現されません。今回の最終学習は固定学習率です。その違いもあるため、最終モデルの精度は別の未使用授業で確認してください。最終モデルを学習に使ったデータで評価した数値は、未知データへの精度とは区別します。

## 実行状態

このワークフローの実装・動作テストを完了しました。本データの全条件比較、CV、最終モデル学習はまだ実行していません。現在のデータから具体的な最良条件・epochが確定したという意味ではありません。

## 既存結果を残してstandardだけ追加する

`standard`は`ordered`と同じ条件に、標準Augmentation（区間で共通の左右反転・切り抜き・色変換など）を加えます。`mild`・`ordered`との比較に使用してください。検証側には適用せず、legacy分類器のAugmentationも引き続き無効です。

```powershell
python scripts/tools/run_accuracy_experiments.py `
  --append --variants standard `
  --output_dir outputs/window_screening --execute
```

保存された元の`experiment_plan.json`のbaselineコマンドから、データ一覧・seed・分割・選択指標・epoch上限・分類器設定を引き継ぎます。追加時はそれらをCLIで指定し直す必要はありません。元のbaselineに複数seedや複数foldがあれば、それぞれにstandardを追加します。

既存のモデル・ログ・実験計画・summary.csvは変更せず、新しい結果を`outputs/window_screening/additions/<日時と識別子>/`へ保存します。新規分だけの`summary.csv`と、旧結果を含む`combined_summary.csv`を出力します。元の分割manifestがある場合は、学習・検証の採用区間が一致することも確認します。同じ条件・seed・foldの完了結果が既にあれば重複実行を拒否します。

`--execute`を省くと追加計画の保存だけです。CV結果に追加する場合も、その結果の出力ディレクトリを`--output_dir`に指定すればfoldを引き継ぎます。通常実行（`--append`なし）は選択した実験を上書きします。

## no_situationでも状況を表示・分析する

今後の比較条件`no_situation`は、`--no-situation-feature`だけを指定し、分類器の学習epochを0にしません。集中度モデルには状況を入力せず、分類器を独立に学習してチェックポイントへ保存します。集中度の学習では独立分類器を生成・実行しないため、分類器を追加したことによって学習乱数列が変わることも避けます。シチュエーション分類器の学習時間は追加されます。

既に学習したno_situationモデルへ、集中度を再学習せずに分類器だけを追加できます。

```powershell
python scripts/tools/train_independent_situation.py `
  --model outputs/window_screening/no_situation_seed42.pt `
  --output models/no_situation_with_classifier.pt `
  --epochs 20
```

入力モデルに保存されたデータ一覧・分割・画像前処理・seedを使い、legacy分類器を学習します。集中度の全重みと元の入力モデルは保持し、新しいファイルへ保存します。20epochは上限の例で、検証ありでは最小CEのepochを保存します。元モデルが検証なしなら分類器も検証なしになります。保存されたデータのパスが現在も利用可能である必要があります。

新しいチェックポイントを動画・リアルタイム推論で指定すると、集中度には状況を入力せず、状況の自動推定・表示・DB記録を行います。`--situation`で手動指定した場合は自動分類を使いません。

`evaluate_labeled_dataset.py --situation-source predicted`では、`predictions.csv`に`predicted_situation`・`situation_confidence`・`situation_probs`を保存します。分類器単独の結果は`situation_metrics.json`・`situation_metrics.csv`・`situation_confusion_matrix.csv`です。集中度のmetricsと分離して集計します。旧no_situationモデルに分類器がなければ、上記の追加学習が必要です。


## 同じ出力先で再実行

通常の`--execute`は、指定したvariant・seed・foldを最初から再学習し、モデル・ログ・学習レポートを上書きします。再開ではありません。summary.csvでは対象行を実行開始時に取り除き、完了した実験から新しい結果へ更新します。選択していない条件のモデル・ログ・集計行は残します。`--final_train`の最終モデルも上書き対象です。学習処理が終了するまで、同名の既存チェックポイントには前回の重みが残る場合があります。
