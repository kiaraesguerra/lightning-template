import torch.nn as nn
import torch
from lrs.base import Base
from lrs.sao_matrix import Ramanujan_Constructions
from lrs.lrs_module import LRSModule


class LowRankSparseInitializer(Base, Ramanujan_Constructions):
    def __init__(
        self,
        model: nn.Module,
        sparse_matrix: str = None,
        threshold: float = 1e-3,
        sparsity: float = None,
        degree: int = None,
        activation: str = "tanh",
        rank: int = None,
    ):
        self.sparse_matrix = sparse_matrix
        self.threshold = threshold
        self.model = model
        self.sparsity = sparsity
        self.degree = degree
        self.activation = activation
        self.rank = rank

    def _low_rank_sparse_module(self, module):
        """Takes in a module of a model then returns an LR (+ S) Module

        Args:
            module (nn.Module): Module which will be turned into an LR (+ S) Module

        Returns:
            nn.Sequential: LR (+ S) Module
        """
        LRS_module = LRSModule(
            in_features=module.weight.shape[1],
            out_features=module.weight.shape[0],
            rank=self.rank,
            sparse_matrix=self.sparse_matrix,
            sparsity=self.sparsity,
            degree=self.degree,
        )

        return LRS_module

    def _sparse_matrix_weights(self, module):
        """Generates a sparse matrix using the Ramanujan Constructions method

        Args:
            module (nn.Module): Module which will be turned into an LR (+ S) Module

        Returns:
            torch.Tensor: Sparse matrix which will be used in the LR (+ S) Module
        """

        constructor = Ramanujan_Constructions(
            height=module.weight.shape[0],
            width=module.weight.shape[1],
            method=self.sparse_matrix,
            sparsity=self.sparsity,
            degree=self.degree,
            activation=self.activation,
        )
        s_weight_matrix, _ = constructor()
        self.s_weight_matrix = s_weight_matrix

        return s_weight_matrix

    def _low_rank_weights_svd(self, module):
        """Generates a low rank sparse matrix using svd"""

        LR = module.weight.reshape(module.weight.shape[0], -1).to("cuda")

        if self.sparse_matrix:
            s_weight_matrix = self._sparse_matrix(module)
            LR = LR - s_weight_matrix

        u, s, v = torch.linalg.svd(LR)
        s_diag = torch.diag_embed(s)
        padded_s = torch.zeros(module.weight.shape[0], module.weight.shape[1])
        padded_s[0 : s_diag.shape[0], 0 : s_diag.shape[1]] = s_diag

        rank = torch.sum(s > self.threshold)
        self.rank = rank
        w = padded_s @ v
        w_weight_matrix = w[0:rank, :]
        u_weight_matrix = u[:, 0:rank]

        return w_weight_matrix, u_weight_matrix, s_weight_matrix

    def _low_rank_weights_explicit(self, module):
        """Generates a low rank sparse matrix explicitly"""

        s_weight_matrix = self._sparse_matrix(module) if self.sparse_matrix else None
        w_weight_matrix = self._ortho_generator(
            module.weights.shape[1], self.rank, self.activation
        )
        u_weight_matrix = self._ortho_generator(
            self.rank, module.weights.shape[0], self.activation
        )

        return w_weight_matrix, u_weight_matrix, s_weight_matrix

    def _low_rank_sparse(self, module):
        w_matrix, u_matrix, s_matrix = (
            self._low_rank_weights_svd(module)
            if not self.rank
            else self._low_rank_weights_explicit(module)
        )
        LRS_module = self._low_rank_sparse_module(module)

        LRS_module.W_layer.weight = nn.Parameter(
            w_matrix.reshape(LRS_module.W_layer.weight.shape)
        )

        LRS_module.U_layer.weight = nn.Parameter(
            u_matrix.reshape(LRS_module.U_layer.weight.shape)
        )

        if self.sparse_matrix:
            LRS_module.S_layer.weight = nn.Parameter(s_matrix)
            torch.nn.utils.prune.custom_from_mask(
                LRS_module.S_layer, name="weight", mask=(s_matrix != 0) * 1
            )

        return LRS_module

    def initialize_low_rank_mixer(self):
        for module_name, module in self.model.mixer_layers.named_modules():
            if isinstance(module, nn.Linear):
                name = list(module_name.split("."))
                setattr(
                    self.model.mixer_layers[int(name[0])].mlp2,
                    name[2],
                    self._low_rank_sparse(module),
                )

            elif isinstance(module, nn.Conv1d):
                name = list(module_name.split("."))
                setattr(
                    self.model.mixer_layers[int(name[0])].mlp1,
                    name[2],
                    self._low_rank_sparse(module),
                )
