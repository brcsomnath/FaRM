from torch import nn


class Net(nn.Module):
    """
    Dynamic MLP network with ReLU non-linearity
    """
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(args.num_layers - 1):
            self.layers.append(
                nn.Linear(args.embedding_size, args.embedding_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(args.embedding_size, args.embedding_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Classifier(nn.Module):
    """
    Classifier for predictions Y_LABELS given input x
    2 hidden layers with a ReLU network
    """
    def __init__(self, args, Y_LABELS):
        super().__init__()
        self.lin1 = nn.Linear(args.embedding_size, args.embedding_size // 2)
        self.lin2 = nn.Linear(args.embedding_size // 2, len(Y_LABELS))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.lin2(self.relu(self.lin1(x)))
