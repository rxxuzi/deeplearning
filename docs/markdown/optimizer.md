# Optimizer

これらは、勾配降下法（SGD）を改良したものである

- **SGD** : 
    各パラメータに対して同じ学習率で勾配方向に更新する基本的な手法です。
- **Momentum** :
SGDに慣性項を加えて、過去の勾配の方向に加速することで、局所最適解を回避したり、谷間を早く抜け出したりする手法
- **AdaGrad** :
各パラメータに対して学習率を適応的に調整することで、勾配が小さいパラメータも十分に更新できるようにする手法
- **RMSProp** :
AdaGradの問題点である学習率の減衰を防ぐために、指数移動平均を用いて勾配の二乗和を計算する手法。
- **Adam** :
MomentumとRMSPropの長所を組み合わせた手法で、各パラメータに対してモーメントと学習率の両方を適応的に調整することで、高速かつ安定的な学習が可能になる手法

~~~python
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []
    def setup(self, target):
        self.target = target
        return self
    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        # 前処理
        for f in self.hooks:
            f(params)
        # パラメータの更新
        for param in params:
            self.update_one(param)
    def update_one(self, param):
        raise NotImplementedError()
    def add_hook(self, f):
        self.hooks.append(f)
~~~


>【最適化手法】SGD・Momentum・AdaGrad・RMSProp・Adamを図と数式で理解しよう<https://kunassy.com/oprimizer/>

> SGD、Momentum、RMSprop、Adam区别与联系 - 知乎. <https://zhuanlan.zhihu.com/p/32488889>

> ニューラルネットワークにおける最適化手法 <https://qiita.com/Fumio-eisan/items/798351e4915e4ba396c2>.

>(3) 【決定版】スーパーわかりやすい最適化アルゴリズム -損失関数><https://qiita.com/omiita/items/1735c1d048fe5f611f80>.
