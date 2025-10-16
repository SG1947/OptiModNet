cf = {
    "image_size": 256,
    "num_channels": 3,
    "num_layers": 12,
    "hidden_dim": 128,
    "mlp_dim": 32,
    "dropout_rate": 0.1,
    "patch_size": 16,
}
cf["num_patches"] = (cf["image_size"]**2) // (cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"] * cf["patch_size"] * cf["num_channels"]
)
cf["num_query_heads"] = 8
cf["num_key_value_heads"] = 4
cf["head_dim"] = cf["hidden_dim"] // cf["num_query_heads"]
