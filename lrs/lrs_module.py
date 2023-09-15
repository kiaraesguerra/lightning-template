import torch.nn as nn


class LRSModule(nn.Module):
    def __init__(
        self,
        module,
        rank: int,
        sparse_matrix: str = None,
        sparsity: float = None,
        degree: float = None,
    ):
        super(LRSModule, self).__init__()

        self.rank = rank
        in_features = module.weight.shape[1]
        out_features = module.weight.shape[0]
        self.sparse_matrix = sparse_matrix
        self.degree = degree
        self.sparsity = sparsity
        self.in_features = in_features
        self.out_features = out_features

        if isinstance(module, nn.Linear):
            self.W_layer = nn.Linear(
                in_features=in_features, out_features=rank, bias=False
            )
            self.U_layer = nn.Linear(
                in_features=rank, out_features=out_features, bias=False
            )
            if self.sparse_matrix:
                self.S_layer = nn.Linear(
                    in_features=in_features, out_features=out_features, bias=False
                )

        elif isinstance(module, nn.Conv1d):
            self.W_layer = nn.Conv1d(
                in_channels=in_features, out_channels=rank, kernel_size=1, bias=False
            )
            self.U_layer = nn.Conv1d(
                in_channels=rank, out_channels=out_features, kernel_size=1, bias=False
            )

            if self.sparse_matrix:
                self.S_layer = nn.Conv1d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=1,
                    bias=False,
                )

    def forward(self, x):
        x1 = self.W_layer(x)
        out = self.U_layer(x1)
        if self.sparse_matrix:
            x2 = self.S_layer(x)
            out += x2
        return out
