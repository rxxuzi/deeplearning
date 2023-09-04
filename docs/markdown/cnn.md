# CNN (Convolutional Neural Networks) のメカニズム

CNNは、画像や音声などのグリッド構造データの自動特徴抽出を得意とするニューラルネットワークの一種である。主に以下の層から構成される。

## 1. Convolutionレイヤ (畳み込みレイヤ)

畳み込みレイヤはCNNの核心となる部分で、ローカルな情報をキャッチする役割がある。これは、入力データに対してフィルタ (またはカーネル) をスライドさせて畳み込みを行うことで実現される。

### 数式:

畳み込み操作は以下の式で表される。
$$
O_{i,j} = \sum_{m} \sum_{n} I_{i+m, j+n} \times K_{m, n}
$$
ここで、$O_{i,j}$ は出力のピクセル、$I_{i+m, j+n}$ は入力のピクセル、$K_{m, n}$ はフィルタの値を表す。

### 例:

![Alt text](../pics/description/image.png)

また、全結合ニューラルネットワークでは、重みパラメータの他に**バイアス**も存在する。

$$
O_{i,j} = \sum_{m} \sum_{n} I_{i+m, j+n} \times K_{m, n} + B
$$

![Alt text](../pics/description/image-1.png)

フィルタの動き:

<img alt="filter" src="../pics/filter.gif" width="400"/>

### 1.2. **パディング (Padding)**

畳み込み操作を行う際に、入力データの周辺に仮のデータ (通常0) を追加すること。これにより、出力データのサイズの縮小を防ぐか、または調整することができる。

本来の入力データと同じ大きさの出力が得られる

![padding](../pics/description/image-2.png)

### 1.3.**ストライド (Stride)**

フィルタをスライドさせる際のステップのサイズ。ストライドが1の場合、フィルタは入力データ上で1ピクセルずつスライドする。ストライドが2の場合は2ピクセルずつとなる。

ストライドを大きくすると出力サイズは小さくなる。また、パディングを大きくすると出力サイズは大きくなる。

![Alt text](../pics/description/image-3.png)

### 1.4.出力サイズの計算

出力サイズ : $O$ ,
入力サイズ : $I$ ,
フィルタ(カーネル) : $K$ ,
ストライド : $S$ ,
パディング : $P$,

とすると、出力サイズは

$$
O = \frac{I + 2P - K}{S} + 1  
$$

で表すことができる

### 1.5. 3階テンソルとしての畳み込み

CNNでは一般的に3階テンソル（3D TENSOR）として画像データを扱う。これは、高さ、幅、そして深さ（チャンネル数）の3つの次元を持っている。例えば、カラー画像はRGBの3つのチャンネルから成るので、3階テンソルとして考えることができる。

![3階テンソル](../pics/description/image-6.png)


畳み込みをブロックで考えると、フィルタは入力データの小さなブロックに適用され、それぞれのブロックの出力が新しい特徴マップのピクセルとなる。

I,K,O,B　をそれぞれ**ブロック**としたもの
![3d](../pics/description/image-5.png)

## 2. Poolingレイヤ (プーリングレイヤ)

プーリングレイヤは、畳み込みによって抽出された特徴のサイズを縮小することで計算量を減らし、特徴の位置感度を低くする役割がある。主なプーリング操作にはMax PoolingやAverage Poolingがある。

### **ダウンサンプリング (Downsampling)**

特徴マップの次元を削減する操作。Max Poolingなどのプーリング操作はダウンサンプリングの一形態として考えられる。

### 例: 

2x2 Max Poolingの場合、2x2の領域の中の最大値がその領域の出力となる。
![2*2 Max Pooling](../pics/description/image-4.png)

## 3. ReLU (Rectified Linear Unit)

ReLUは、非線形性をネットワークに導入するための活性化関数の一つだ。これは、負の入力値に対しては0を、正の入力値に対してはそのままの値を出力するというシンプルな関数だが、深いネットワークでの学習を助ける特性を持つ。


### 3.1. 数式:

ReLU関数は以下の式で表される。
$$
f(x) = \max(0, x)
$$

### **局所コントラスト正規化 (Local Contrast Normalization)**

畳み込みによって得られた特徴マップの各要素を、その近傍の要素の平均や標準偏差を用いて正規化する操作。これにより、特徴の局所的なコントラストが強調される。

## 4. バッチ正規化 (Batch Normalization)

バッチ正規化は、ミニバッチごとの入力分布を正規化することで、学習を安定化させ、高速化する手法である。
具体的には、各ミニバッチの平均と分散を用いて正規化を行い、さらにスケール変換と平行移動を可能にする学習可能なパラメータを持つ。

ミニバッチのサンプルを $n=(1,2\dots N)$で表し、畳み込み層のチャンネルcを構成する 
$W \times H$ 内の位置 $(i,j)$のユニットへの層入力を 
${y}_{ijc}^{(n)}$とする時、平均は以下のように表せる。

$$
\mu_C = \frac{1}{NWH} \sum_{i,j,n} {u}_{ijc}^{(n)}
$$

次のように正規化する
$$
\hat{u}_{ijc}^{(n)} = \gamma_C \frac{{u}_{ijc}^{(n)} - \mu_C}{\sqrt{\sigma_C^2 + \epsilon}} + \beta_C

$$

ここで、$\gamma$ と $\beta$ は学習可能なパラメータ(学習によって定めた定数)であり、

$\mu_B$ と $\sigma_B^2$ はミニバッチの平均と分散を示す。

### 4.2. 流れ

<img alt="Original pic" src="../pics/description/Original.png" width="120"/> <br>

<img alt="R" height="100" src="../pics/description/Original_OnlyRedComponent.png"/>
<img alt="G" height="100" src="../pics/description/Original_OnlyGreenComponent.png"/>
<img alt="B" height="100" src="../pics/description/Original_OnlyBlueComponent.png"/>

### 4.1. ミニバッチ処理 (Mini-batch processing)

大量のデータセットに対する学習を効率化するため、一度に小さなサブセット（ミニバッチ）を取り出して学習を行う方法。ミニバッチごとの勾配の平均を用いてパラメータの更新を行う。

## 6. 学習

CNNの学習は、一般的なニューラルネットワークの学習と同様に、バックプロパゲーションを使用して行われる。バックプロパゲーションは、予測誤差を最小化するためのパラメータの調整を行うアルゴリズムであり、損失関数の勾配を計算して、その勾配情報を用いてパラメータを更新する。

最適化手法には、SGD (Stochastic Gradient Descent) やその派生であるMomentum, AdaGrad, RMSprop, Adamなどがある。


---

## 参考:

>一文看懂variant convolutions
> <https://blog.csdn.net/WANGWUSHAN/article/details/103575060>

>深層学習 改訂第2版　岡谷貴之

