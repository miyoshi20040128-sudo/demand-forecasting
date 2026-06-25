## config.toml

### `[model]`：モデル構造の設定


#### `seq_len`

```toml
seq_len = 30
```

モデルに入力する過去の時系列長を表す。
ここでは、過去30日分のデータを使って予測を行う。

#### `pred_len`

```toml
pred_len = 1
```

モデルが予測する未来の時系列長を表す。
ここでは、未来1日分の販売数を予測する。

#### `d_model`

```toml
d_model = 64
```

Transformer内部で使う特徴表現の次元数である。
値を大きくすると表現力は上がるが、計算量も増える。

#### `n_heads`

```toml
n_heads = 4
```

Multi-Head Attention のヘッド数を表す。
複数の注意機構を並列に使うことで、異なる観点から時系列の関係を捉える。

#### `num_layers`

```toml
num_layers = 2
```

Transformer Encoder Layer の層数を表す。
値を大きくするとモデルは深くなるが、過学習や計算量増加の可能性もある。

#### `dropout`

```toml
dropout = 0.1
```

過学習を抑えるための dropout 率である。
ここでは、学習中に一部のニューロンを10%の確率で無効化する。

---

### `[train]`：学習条件の設定


#### `batch_size`

```toml
batch_size = 32
```

1回のパラメータ更新で使うデータ数を表す。
ここでは、32サンプルごとに勾配を計算してモデルを更新する。

#### `epochs`

```toml
epochs = 100
```

最大の学習回数を表す。
ただし、後述する `patience` によって早期終了する場合がある。

#### `learning_rate`

```toml
learning_rate = 0.001
```

学習率を表す。
値が大きすぎると学習が不安定になり、小さすぎると学習に時間がかかる。

#### `patience`

```toml
patience = 20
```

Early Stopping の設定である。
検証誤差が20エポック連続で改善しなかった場合、学習を停止する。

#### `seed`

```toml
seed = 42
```

乱数シード


指定できる損失関数は以下である。

```toml
loss = "mse"
loss = "mae"
loss = "huber"
loss = "dilate"
loss = "derivative"
```

---

### `[loss]`：損失関数ごとのパラメータ

```toml
[loss]
alpha = 0.8
gamma = 0.1
huber_delta = 1.0
derivative_alpha = 1.0
derivative_beta = 0.1
```

#### DILATE Loss 用パラメータ

```toml
alpha = 0.8
gamma = 0.1
```

DILATE Loss は、時系列の形のずれと時間方向のずれを考慮する損失関数である。

`alpha` は、Shape Loss と Temporal Loss の重みを決める。

```text
alpha が大きいほど、時系列の形を重視する
alpha が小さいほど、時間方向のずれを重視する
```

ここでは、

```toml
alpha = 0.8
```

としているため、時間方向のずれよりも、時系列の形状をやや強く重視する設定である。

`gamma` は Soft-DTW の滑らかさを調整するパラメータである。

```toml
gamma = 0.1
```

`gamma` が小さいほど通常の DTW に近くなり、値が大きいほど滑らかな損失になる。

---

#### Huber Loss 用パラメータ

```toml
huber_delta = 1.0
```

Huber Loss は、MSE と MAE の中間的な性質を持つ損失関数である。

`huber_delta` は、MSE的に扱う範囲と MAE的に扱う範囲を切り替える閾値である。

```text
|error| <= delta の場合: MSE のように二乗誤差として扱う
|error| > delta の場合: MAE のように絶対誤差として扱う
```

本コードでは目的変数を標準化しているため、

```toml
huber_delta = 1.0
```

は標準化後の誤差が1程度までは MSE 的に扱い、それ以上の誤差は MAE 的に扱うという意味になる。

---

#### Derivative Loss / CONTIME風 Loss 用パラメータ

```toml
derivative_alpha = 1.0
derivative_beta = 0.1
```

Derivative Loss は、値そのものの誤差に加えて、時系列の変化量の誤差も考慮する損失関数である。
CONTIME の考え方を、Transformer の出力系列に対して差分で近似している。

損失関数は次のように表される。

```text
loss = derivative_alpha * MSE(y_pred, y_true)
     + derivative_beta  * MSE(diff(y_pred), diff(y_true))
```

`derivative_alpha` は、予測値そのものの誤差をどれだけ重視するかを表す。
`derivative_beta` は、時系列の増減パターンをどれだけ重視するかを表す。

ここでは、

```toml
derivative_alpha = 1.0
derivative_beta = 0.1
```

としており、値そのものの誤差を中心にしつつ、増減のズレにも少しペナルティを与える設定になっている。

---

### `[compare]`：複数損失関数の一括比較

```toml
[compare]
loss_names = ["mse", "mae", "huber", "dilate", "derivative"]
```

`compare` セクションでは、1回の実行で比較する損失関数を指定する。

この設定では、以下の5つの損失関数を順番に実行する。

```text
mse
mae
huber
dilate
derivative
```

一括実行用の `main.py` を実行すると、`loss_names` に書かれた損失関数が順番に学習され、それぞれの結果が `output/` 以下に保存される。

出力例は以下のようになる。

```text
output/
  mse/
  mae/
  huber/
  dilate/
  derivative/
  loss_comparison.csv
  loss_comparison_with_graphs.jpg
```



`loss_comparison.csv` には、各損失関数の MAE、RMSE、MAPE などの評価結果が保存される。
`loss_comparison_with_graphs.jpg` には、各損失関数による時系列予測結果を比較した画像が保存される。
