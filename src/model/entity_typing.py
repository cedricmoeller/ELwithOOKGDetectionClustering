from torch import sigmoid
from torch.nn import Linear, Module


class EntityTyper(Module):
    def __init__(self, input_dim, num_types, normalize=True):
        super().__init__()
        self.linear = Linear(input_dim, num_types)
        self.normalize = normalize


    def forward(self, x):
        out = self.linear(x)
        if self.normalize:
            out = sigmoid(out)
        return out