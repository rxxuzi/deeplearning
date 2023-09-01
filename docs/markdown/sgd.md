# Stochastic Gradient Descent

ニューラルネットワークの学習は損失関数の最小化に帰着される。
順伝搬型ネットワークの学習は、訓練データ

$$
D = {(x_1 , d_1),(x_2,d_2) \dots (x_n,d_n)}
$$

に対して計算される損失関数 $E(w)$ を$w$ (パラメータ)について最小化する事に帰着する。

## 1.確率的勾配降下法 (Stochastic Gradient Descent: SGD)

損失関数 $E$ とパラメータ w の関係を考えるとき、損失の勾配は以下のように表せる:

$$
\nabla E(w) = \frac{\partial E}{\partial w}
$$

$w$ を負の勾配方向 $( - \nabla E )$ に少しだけ動かす。

つまり、現在の重みを $w_{\text{new}}$ , 動かした後の重みを $w_{\text{old}}$ とするとSGDのパラメータの更新は次のようになる:

$$
w_{\text{new}} = w_{\text{old}} - \alpha \nabla E(w_{\text{old}}) 
$$

ここで $\alpha$ は $w$ の更新量の大きさを定める定数で、**学習率(Learning rate)** と呼ぶ。

## 2.ミニバッチ (Mini-batch)

ミニバッチ $B$ に対する平均勾配は以下のように計算する:

$$
\nabla_B E(w) = \frac{1}{|B|} \sum_{i \in B} \frac{\partial E_i}{\partial w}
$$

## 3.モメンタム (Momentum)

モメンタムを用いた更新は次の式で表せる:

$$
v_{\text{new}} = \beta v_{\text{old}} + (1 - \beta) \nabla E(w_{\text{old}}) 
$$

$$
w_{\text{new}} = w_{\text{old}} - \alpha v_{\text{new}}
$$

$v$ は過去の勾配の移動平均、
$\beta$ はモメンタム項（通常0.9付近の値）。

## 4.過剰適合 (Overfitting)

訓練データ $D$ の損失 $E_{\text{train}}(D)$ が小さいとき、テストデータの損失 $E_{\text{test}}(D')$ が大きい場合、過剰適合とみなす。

## 5.バイアス (Bias)

ニューラルネットワークの線形変換 $z = Wx + b$ において、$b$ はバイアス。
