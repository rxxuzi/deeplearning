# ベクトルの内積・行列の積

## ベクトルの積

2つのベクトル$\vec{a} = (a_1 , a_2, \cdots , a_n )$ と $\vec{b} = (b_1 , b_2, \cdots , b_n )$があると仮定したとき、この時ベクトルの内積は次の式になる

$\vec{a}\vec{b} =a_1b_1 + a_2b_2 + \cdots + a_nb_n$

## 行列の積

$$A = \begin{pmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\
a_{21} & a_{22} & \ldots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \\
\end{pmatrix}$$
$$B =
\begin{pmatrix}
b_{11} & b_{12} & \ldots & b_{1p} \\
b_{21} & b_{22} & \ldots & b_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
b_{n1} & b_{n2} & \ldots & b_{np} \\
\end{pmatrix}$$

$$C = A \cdot B = \begin{pmatrix}c_{11} & c_{12} & \ldots & c_{1p} \\c_{21} & c_{22} & \ldots & c_{2p} \\\vdots & \vdots & \ddots & \vdots \\c_{m1} & c_{m2} & \ldots & c_{mp} \\\end{pmatrix}$$

ここで、各要素 $c_{ij}$ は次のように計算される：

$c_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j} + \ldots + a_{in}b_{nj} = \sum_{k=1}^{n} a_{ik}b_{kj}$

つまり、新しい行列Cの各要素は、行列Aのi行と行列Bのj列の対応する要素を掛け合わせて総和を取ることで計算される。

以上が行列の積の計算方法である。これに従って適切な値を代入して計算すれば、行列の積を求めることができる。

また、行列やベクトルを使った計算は**形状**に注意する必要がある.

例えば$A$が3×2行列,$B$が2×4行列の場合、$C$は3×4行列になる

$C$は$A$の行数と$B$の列数から構成されている

## 行列の積の逆伝播

各行列の形状は$x = 1 \times D, W = D \times H , y = 1 \times H$
とする.

$y = xW$

という計算を考える。
**$y$が何らかの計算によって最終的に$L$というスカラが出力される**この$L$の各変数に関する微分を逆伝播で求める

この時、$x$の$i$番目の要素に関する微分$\frac{\partial L}{\partial x_i}$は以下の式で求められる

$\frac{\partial L}{\partial x} = \sum_{j}\frac{\partial L}{\partial y_j}\frac{\partial y_j}{\partial x_i}$

上記の式は$x_i$を変化させたときに、$L$がどれだけ変化するのかという**変化の割合**を表している。ここで、$x_i$を変化させた時はベクトル$y$のすべての要素も変化する。そして、$y$の各要素の変化を通じて、最終的に$L$が変化することになる。
そのため、$x_i$から$L$にいたるチェインルールの経路は複数あり、その総和が$\frac{\partial L}{\partial x_i}$となる。

またこの式は簡略化できる。

$y_j = x_1W_{1j} + x_2W_{2j} +x_3W_{3j} + \cdots + x_HW_{Hj}$

より$\frac{\partial y_j}{\partial x_i} = W_{ij}$となる

$\frac{\partial L}{\partial x}
= \sum_{j}\frac{\partial L}{\partial y_j}\frac{\partial y_j}{\partial x_i}
= \sum_{j}\frac{\partial L}{\partial y_j} \vec{W_{ij}}
$

また上の式より、$\frac{\partial L}{\partial x_i}$は
**ベクトル$\frac{\partial L}{\partial \vec{y}}$** と **$\vec{W}$の$i$行目のベクトル**の**内積**によって求めることができる。この関係から次の式が導ける

$\frac{\partial L}{\partial \vec{x}} = \frac{\partial L}{\partial \vec{y}}\vec{W}^T$となる.
