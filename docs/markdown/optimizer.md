了解しました。以下はOptimizerに関するレポートです。

---

# Optimizerについて

深層学習におけるOptimizerは、モデルの重みを更新して学習を進める際のアルゴリズムを指す。以下、主要なOptimizerとその特徴について詳しく解説します。

## 1. SGD (確率的勾配降下法: Stochastic Gradient Descent)

SGDは最も基本的なOptimizerであり、各更新で1つのサンプルまたはミニバッチの勾配を計算して重みを更新する。

**重みの更新式:**
$$
w_{t+1} = w_t - \eta \nabla E(w_t)
$$
ここで、\( \eta \) は学習率、\( \nabla E(w_t) \) は重み \( w_t \) における損失 \( E \) の勾配を示す。

## 2. Momentum

Momentumは、過去の勾配の情報を利用して局所的な最適解に収束するのを防ぎ、収束を加速するための手法。

**重みの更新式:**
$$
v_{t+1} = \mu v_t - \eta \nabla E(w_t)
$$
$$
w_{t+1} = w_t + v_{t+1}
$$

ここで、\( \mu \) はモメンタム係数、\( v_t \) は時刻 \( t \) における勾配の移動平均を表す。

## 3. AdaGrad

AdaGradは、学習率を各パラメータごとに適応的に調整するOptimizer。

**重みの更新式:**
$$
G_{t+1} = G_t + \nabla E(w_t) \odot \nabla E(w_t)
$$
$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} \odot \nabla E(w_t)
$$

ここで、\( G_t \) は勾配の二乗の累積、\( \odot \) は要素ごとの積、\( \epsilon \) は0で割るのを防ぐための小さな定数。

## 4. RMSProp

RMSPropはAdaGradの問題を改善するために提案されたもので、過去のすべての勾配を均等に考慮するのではなく、最近の勾配の情報を主に利用する。

**重みの更新式:**
$$
G_{t+1} = \rho G_t + (1 - \rho) \nabla E(w_t) \odot \nabla E(w_t)
$$
$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} \odot \nabla E(w_t)
$$

ここで、\( \rho \) は減衰率。

## 5. Adam

AdamはMomentumとRMSPropのアイディアを組み合わせたOptimizer。移動平均と移動分散をそれぞれ計算し、それを使って重みを更新する。

**重みの更新式:**
$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla E(w_t)
$$
$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2) \nabla E(w_t) \odot \nabla E(w_t)
$$
$$
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}
$$
$$
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
$$
$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1} + \epsilon}} \odot \hat{m}_{t+1}
$$

ここで、\( \beta_1 \) と \( \beta_2 \) はハイパーパラメータ、

\( m_t \) と \( v_t \) はそれぞれ勾配の移動平均と移動分散を表す。

---

以上が、Optimizerに関する主要な手法とその特徴についてのレポートです。これらのOptimizerは、様々な深層学習のタスクにおいて利用され、モデルの学習の効率や性能に大きな影響を与える要素となっています。


>【最適化手法】SGD・Momentum・AdaGrad・RMSProp・Adamを図と数式で理解しよう<https://kunassy.com/oprimizer/>

> SGD、Momentum、RMSprop、Adam区别与联系 - 知乎. <https://zhuanlan.zhihu.com/p/32488889>

> ニューラルネットワークにおける最適化手法 <https://qiita.com/Fumio-eisan/items/798351e4915e4ba396c2>.

>(3) 【決定版】スーパーわかりやすい最適化アルゴリズム -損失関数><https://qiita.com/omiita/items/1735c1d048fe5f611f80>.
