# Augmentation none / standard の評価確認（2026年10月9日）

## 結論

学習結果は異なる。提供された2つのmetrics JSONはバイト単位でも同一で、両方ともnoneモデルの再評価結果と全指標で一致する。standardモデルの再評価結果は異なる。評価で同じモデルを指定した、またはJSONのコピー・保存時に取り違えた可能性が高い。元の評価コマンド／run_configが未提供のため、どちらの経路だったかは未確定。

## モデルと保存設定

| ファイル | 保存されたaugmentation | 最良検証MSE |
| --- | --- | ---: |
| outputs/pre_test_fusion_transformer_aug_auto_situation.pt | none | 1.226706 |
| outputs/pre_test_fusion_transformer_aug_situation_202610090028.pt | standard | 1.546692 |

ファイル名にaugが含まれていても、最初のモデルの設定はnone。保存cfgの差はaugmentationのみで、train/validation区間リストも一致。集中度モデルの157個の重み・バッファはいずれも異なり、シチュエーションモデルの重みと学習履歴も異なる。

現在のコードではCLIからCfgへaugmentationを渡し、学習Datasetのself.augmentで有効／無効を切り替える。同じ実区間・seedで入力を確認したところ、noneは変換無効、standardは有効となり、画像Tensorが異なることを確認した。評価時はどちらもaugmentationを適用しない。

## 再評価

対象: 20260227_unnan_nishi_5-1_1、評価者tadano、シチュエーション自動判定、人物除外後2,391区間。同じ入力を各モデルへ渡し、モデルの通常のforward経路で評価した。評価者sampleからtadanoへの変更は名前のみで、採点値は変わっていない。

| 指標 | none | standard |
| --- | ---: | ---: |
| exact_accuracy | 72.27% | 60.02% |
| ±1以内 | 94.35% | 88.00% |
| MAE | 0.4091 | 0.6240 |
| RMSE | 0.7234 | 1.0322 |

今回の学習済みdatasetを多く含む評価ではnoneが良い。未知の授業や撮影条件への効果までは、この結果では判断できない。

元のJSON・チェックポイントは上書きせず、正しい再評価結果をoutputs/augmentation_audit_20261009/metrics_none.json、metrics_standard.jsonへ保存した。同フォルダのcomparison.jsonに使用モデルと設定、predictions.csvに区間ごとの予測を記録した。

取り違えを確認しやすくするため、通常の評価ツールにも、読み込んだモデルの絶対パスと保存されたtraining_augmentationを開始時に表示し、run_config.jsonへ記録する処理を追加した。構文チェック成功。
