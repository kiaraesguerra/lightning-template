from lrs.lrs_init import LowRankSparseInitializer


def get_ls_init(model, args):
    initializer = LowRankSparseInitializer(
        model,
        sparse_matrix=args.sparse_matrix,
        threshold=args.threshold,
        sparsity=args.sparsity,
        degree=args.degree,
        activation=args.activation,
        rank=args.rank,
    )

    initialized_model = initializer.initialize_low_rank_mixer().to("cuda")

    return initialized_model
