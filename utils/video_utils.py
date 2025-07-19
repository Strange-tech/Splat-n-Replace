import cv2
import os

def extract_frames(video_path, output_dir, target_width=1080, num_frames=200):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算等间隔的帧索引
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    idx = 0
    for frame_id in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id == frame_indices[idx]:
            # 缩放图像
            h, w = frame.shape[:2]
            scale = target_width / w
            target_height = int(h * scale)
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # 保存为 JPG 格式
            out_path = os.path.join(output_dir, f"frame{idx:04d}.jpg")
            # 可选：设置 JPEG 压缩质量（默认是 95）
            cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            idx += 1

            if idx >= num_frames:
                break

    cap.release()
    print(f"提取完毕，共保存 {idx} 张图像到 {output_dir}")

# 示例调用
extract_frames("/root/autodl-tmp/data/toy/1755557_Housing_Crisis_Housing_Shortage_3840x2160.mp4", "/root/autodl-tmp/data/toy/images")