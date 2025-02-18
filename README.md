## 概要
異常検出モデルの学習用スクリプトです．

## ファイル・フォルダ構成
### Pythonスクリプト
- `learn_videomae.py`: 動画を入力とするクラス分類モデルの学習用スクリプト
- `learn_audio.py`: 音声を入力とするクラス分類モデルの学習用スクリプト
- `learn_multimodal.py`: 動画と音声を入力とするクラス分類モデルの学習用スクリプト
- `learn_metric_mae.py`: 動画を入力とする深層距離学習モデルの学習用スクリプト
- `learn_metric_vggish.py`: 音声を入力とする深層距離学習モデルの学習用スクリプト
- `learn_metric_mm.py`: 動画と音声を入力とする深層距離学習モデルの学習用スクリプト
- その他Pythonスクリプト: モデルの定義等に必要です．必ず学習用スクリプトと同じ階層に配置してください．
### その他フォルダ
- `data/`: 学習データ作成のためのラベルが定義されたCSVファイルが入ってます
- `checkpoints/`: 作成されたモデルのチェックポイントが入ってます
- `embeddings/`: [リアルタイムな推論のためのスクリプト](https://github.com/m0chi1216/anomaly_detection/tree/main)の利用時に必要なファイルが入ってます

## 環境
- Python 3.10.13
- パッケージ: `requirements.txt`を用いてインストールしてください

## 注意
- マルチモーダルモデルの学習には，同じシードで学習された動画モデルと音声モデルが必要です
  - 例えば，`learn_metric_mm.py`で，seed=42で学習する場合，まず，`learn_metric_mae.py`及び`learn_metric_vggish.py`でseed=42で学習を行い，チェックポイントを作成する必要があります

## リアルタイム異常検出を行う手順
- 以下の2つのリポジトリを実行環境でクローンしてください
  - [interaction_video_server](https://github.com/m0chi1216/interaction_video_server)
  - [realtime_detection](https://github.com/m0chi1216/realtime_detection)
- `realtime_detection/checkpoints/` 下に，`./checkpoints/mm-metric/rank-96-best.pth`を配置してください
- `realtime_detection/embeddings/` 下に，`./embeddings/` 下の3つのファイルを全て配置してください
- 各リポジトリのREADMEに従って実行してください