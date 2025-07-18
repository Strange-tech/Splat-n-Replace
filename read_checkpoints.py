import torch

checkpoint = torch.load("/chkpnt10000.pth")

features_dc_offsets = checkpoint[0][-2]

feature_rest_offsets = checkpoint[0][-1]

for k,v in features_dc_offsets.items():
    # print(k, v)
    for dense_tensor in v:
        total_elements = dense_tensor.numel()
        zero_count = (dense_tensor == 0).sum().item()
        print(zero_count, total_elements)
        # sparse_tensor = dense_tensor.to_sparse()
        # dense_size = dense_tensor.element_size() * dense_tensor.numel()
        # indices_size = sparse_tensor.indices().element_size() * sparse_tensor.indices().numel()
        # values_size = sparse_tensor.values().element_size() * sparse_tensor.values().numel()
        # sparse_size = indices_size + values_size
        # print(f"稠密张量大小: {dense_size} bytes")
        # print(f"稀疏张量大小: {sparse_size} bytes")
        # print(f"压缩比例: {(sparse_size / dense_size):.2%}")
