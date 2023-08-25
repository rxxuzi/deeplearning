from dezero import Layer
from dezero import utils

import dezero.functions as F
import dezero.layers as L


# =============================================================================
# Model / Sequential / MLP
# =============================================================================

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)




class MLP(Model):
    def __init__(self, fc_output_sizes):
        super().__init__()
        self.fc_layers = []
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.fc_layers.append(layer)

    def forward(self, x):
        for l in self.fc_layers:
            x = F.sigmoid(l(x))
        return x

def softmax_simple(x):
    x = utils.as_variable(x)
    y = F.softmax(x)
    return y