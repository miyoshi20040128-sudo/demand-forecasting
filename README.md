# 需要予測（iTransformer）

販売数と天気の時系列から、iTransformer で将来の販売数を予測する実験です。同じデータ・同じシードで、MSE / MAE / Huber / DILATE / Derivative の損失を比較します。

## ディレクトリ

コードは `same_seedmulti/` にあります。

| パス | 内容 |
| --- | --- |
| `same_seedmulti/src/model.py` | iTransformer |
| `same_seedmulti/src/losses.py` | 損失関数 |
| `same_seedmulti/src/dataset.py` | データの読み込みと窓切り |
| `same_seedmulti/src/train.py` | 学習と評価 |
| `same_seedmulti/main.py` | 損失を順に学習し、比較レポートを作る |
| `same_seedmulti/config.toml` | データパス、モデル、学習条件 |
| `same_seedmulti/data/` | `sales.csv` と `weather.csv` |
| `same_seedmulti/output/` | 予測、指標、損失曲線、モデル重み |

設定項目の説明は [same_seedmulti/readme.md](same_seedmulti/readme.md) にあります。

## 実行

Python 3.11 以上が必要です（`tomllib` を使います）。

```bash
pip install torch pandas numpy matplotlib
cd same_seedmulti
python main.py
```

`config.toml` の `[compare] loss_names` に書いた損失が順番に学習されます。

## 出力

各損失の結果は `output/<loss>/` に保存されます。比較表は `output/loss_comparison.csv`、予測グラフの比較は `output/loss_comparison_with_graphs.jpg` です。
