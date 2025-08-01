import numpy as np
import torch
import argparse
import os 

# 读取npy文件的函数
def load_npy_file(file_path):
    try:
        data = np.load(file_path)
        print(f"成功加载文件: {file_path}")
        return data
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None

# 示例用法
def main():
     
    parser = argparse.ArgumentParser(description="Translate .npy to .pt of masks.")
    parser.add_argument("--input_dir", type=str, required=True, help="input .npy dir")
    parser.add_argument("--output_dir", type=str, default="./output", help="output .pt dir")

    args = parser.parse_args()
    # folder_path = "/workspace/Grounded-SAM-2/outputs/mask_data/"
    folder_path = args.input_dir
    output_path = args.output_dir

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created folder: {output_path}")
        
    # 遍历文件夹中的所有.npy文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            id = filename.split('.')[0].split('mask_frame')[-1]
            print(f"Processing file: {filename}, ID: {id}")
            file_path = os.path.join(folder_path, filename)
            data = load_npy_file(file_path)
            if data is not None:
                print(f"文件: {filename}, 形状: {data.shape}")
                # 这里可以添加对data的进一步处理
                mask_map = {}
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if data[i, j] != 0:
                            if data[i, j] not in mask_map:
                                mask_map[data[i, j]] = np.zeros_like(data, dtype=bool)
                            mask_map[data[i, j]][i][j] = True
                masks = []
                for key, value in mask_map.items():
                    masks.append(value)
                masks = np.array(masks)
                print("masks shape:", masks.shape)
                # torch.save(torch.from_numpy(masks), f'/workspace/SegAnyGAussians/data/replica_data/Replica_colmap/sampled_room0/grounded_sam_masks/frame{id}.pt')
                torch.save(torch.from_numpy(masks), f'{output_path}/frame{id}.pt')

if __name__ == "__main__":
    main()