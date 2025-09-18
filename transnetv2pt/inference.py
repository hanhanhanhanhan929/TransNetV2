import os
import cv2
import torch
import numpy as np
import logging
from tqdm import tqdm
from transnetv2_pytorch import TransNetV2
import subprocess
def seconds_to_ffmpeg_time(seconds):
    """将秒数转换为 FFmpeg 的 HH:MM:SS.ms 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"
def extract_frames_with_opencv(video_path: str, target_height: int = 27, target_width: int = 48, show_progressbar: bool = False):
    """
    Extracts frames from a video using OpenCV with optional CUDA support and progress tracking.
    """
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Initialize progress bar
    progress_bar = tqdm(total=total_frames, desc="Extracting frames", unit="frame") if show_progressbar else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (target_width, target_height))
        frames.append(frame_resized)
        if progress_bar:
            progress_bar.update(1)

    cap.release()
    if progress_bar:
        progress_bar.close()
    logger.info(f"Extracted {len(frames)} frames")
    return np.array(frames)

def input_iterator(frames):
    """
    Generator that yields batches of 100 frames, with padding at the beginning and end.
    """
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded_inputs = np.concatenate(
        [start_frame] * no_padded_frames_start +
        [frames] +
        [end_frame] * no_padded_frames_end, 0
    )

    ptr = 0
    while ptr + 100 <= len(padded_inputs):
        out = padded_inputs[ptr:ptr + 100]
        ptr += 50
        yield out[np.newaxis]

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    """
    Converts model predictions to scene boundaries based on a threshold.
    """
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)

def predict_raw(model, video, device=torch.device('cuda:0')):
    """
    Performs inference on the video using the TransNetV2 model.
    """
    model.to(device)
    with torch.no_grad():
        predictions = []
        for inp in input_iterator(video):
            video_tensor = torch.from_numpy(inp).to(device)
            single_frame_pred, all_frame_pred = model(video_tensor)
            single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
            all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
            predictions.append(
                (single_frame_pred[0, 25:75, 0], all_frame_pred[0, 25:75, 0]))
        single_frame_pred = np.concatenate([single_ for single_, _ in predictions])
        return video.shape[0], single_frame_pred

def predict_video(video_path: str, device: str = 'cuda', show_progressbar: bool = False):
    """
    Detects shot boundaries in a video file using the TransNetV2 model.
    """
    # Determine device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    frames = extract_frames_with_opencv(video_path, show_progressbar=show_progressbar)
    _, single_frame_pred = predict_raw(model, frames, device=device)
    scenes = predictions_to_scenes(single_frame_pred)
    logger.info(f"Detected {len(scenes)} scenes")
    return scenes
# --- 新增的核心功能：视频切割函数 ---
def split_video_into_scenes_ffmpeg(original_video_path: str, scenes: np.ndarray, output_dir: str):
    """
    使用 FFmpeg 根据场景的起止帧号，将原始视频切割成多个子视频。
    这种方法通过流复制实现，速度极快且无损，并能避免编码器问题。

    :param original_video_path: 原始视频文件的路径。
    :param scenes: 一个Numpy数组，每行包含[start_frame, end_frame]。
    :param output_dir: 保存切割后视频的文件夹路径。
    """
    # 检查 FFmpeg 是否存在
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("错误: FFmpeg 未安装或未在系统 PATH 中。请先安装 FFmpeg。")
        return

    # 检查输出目录是否存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 打开原始视频文件以获取信息
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        logger.error(f"错误: 无法打开原始视频文件 {original_video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release() # 获取信息后即可释放
    
    if fps == 0:
        logger.error(f"错误: 无法获取视频 {original_video_path} 的 FPS。")
        return

    logger.info(f"原始视频信息: FPS={fps:.2f}")
    
    # 2. 遍历每个场景并使用 FFmpeg 进行切割
    for i, (start_frame, end_frame) in enumerate(scenes):
        scene_number = i + 1
        # 注意：输出文件扩展名最好与原始视频保持一致，这里假设为 .mp4
        output_path = os.path.join(output_dir, f"scene_{scene_number:03d}.mp4")
        
        logger.info(f"正在处理场景 {scene_number}/{len(scenes)}: 帧 {start_frame} 到 {end_frame}")
        
        # 3. 计算起始和结束时间戳
        start_time = start_frame / fps
        end_time = (end_frame + 1) / fps # +1 是为了包含最后一帧
        
        start_time_str = seconds_to_ffmpeg_time(start_time)
        end_time_str = seconds_to_ffmpeg_time(end_time)
        
        # 4. 构建并执行 FFmpeg 命令
        command = [
            'ffmpeg',
            '-y',  # 如果输出文件已存在则自动覆盖
            '-i', original_video_path,  # 输入文件
            '-ss', start_time_str,      # 起始时间
            '-to', end_time_str,        # 结束时间
            '-c', 'copy',               # 关键！复制流，不重新编码
            '-avoid_negative_ts', 'make_zero', # 避免时间戳问题
            output_path                 # 输出文件
        ]
        
        try:
            # 使用 subprocess.run 执行命令
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"场景 {scene_number} 已保存至: {output_path}")
        except subprocess.CalledProcessError as e:
            # 如果 FFmpeg 执行失败，打印错误信息
            logger.error(f"处理场景 {scene_number} 时 FFmpeg 出错:")
            logger.error(f"FFmpeg Stderr: {e.stderr}")

    # 5. 释放原始视频的读取器
    cap.release()
    logger.info("所有场景切割完成！")

if __name__=='__main__':
    # Initialize logger
    logger = logging.getLogger(__name__)

    #Initialize TransNetV2 model
    model = TransNetV2()
    state_dict = torch.load(
       '/data/hanrui/TransNetV2/transnetv2pt/transnetv2-pytorch-weights.pth'
    )
    msg=model.load_state_dict(state_dict)
    model.eval()
    target = "/data/hanrui/data/demo.mp4"
    output_folder = "/data/hanrui/TransNetV2/output_scenes" # 建议使用一个新文件夹
    detected_scenes = predict_video(str(target), device='cuda', show_progressbar=True)
    print(detected_scenes)
    # --- 3. 根据检测结果切割视频 ---
    #split_video_into_scenes_ffmpeg(target,detected_scenes,output_folder)
   
    