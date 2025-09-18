import base64
from openai import OpenAI
from decord import VideoReader
import subprocess
from decord import VideoReader, cpu
import os

def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

#加载视频，判断视频是否需要抽帧
def preprocess_video(video_path, duration_threshold=1200, output_fps=10,target_bitrate='1000k'):
    """
    预处理视频。如果视频时长超过阈值，则进行抽帧并生成新视频。

    Args:
        video_path (str): 原始视频文件路径。
        duration_threshold (int): 时长阈值（秒）。默认为 1200 秒 (20 分钟)。
        output_fps (int): 生成的新视频的帧率。默认为 10。

    Returns:
        str: 处理后的视频文件路径（可能是新文件或原文件）。
    """
    print(f"正在检查视频: {video_path}")
    
    try:
        # 使用 decord 高效读取视频信息
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration_seconds = total_frames / fps/2
        
        print(f"视频信息 -> 时长: {duration_seconds:.2f} 秒, 帧率: {fps:.2f}, 总帧数: {total_frames}")

        # 1. 检查视频时长是否超过阈值
        if duration_seconds <= duration_threshold:
            print("视频时长未超过阈值，无需抽帧处理。")
            return video_path

        # 2. 如果超长，开始抽帧
        print(f"视频时长超过 {duration_threshold / 60} 分钟，开始抽帧...")
        
        # 定义输出的新视频文件名
        base, ext = os.path.splitext(video_path)
        sampled_video_path = f"{base}_sampled.mp4"
        
        # 获取视频的宽度和高度用于写入
        sample_frame = vr[0].asnumpy()
        height, width, _ = sample_frame.shape
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-framerate', str(output_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-b:v', target_bitrate,
            '-pix_fmt', 'yuv420p',
            sampled_video_path
        ]
        ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        num_seconds_to_sample = int(duration_seconds)
        for i in range(num_seconds_to_sample):
            frame_index = int(i * fps)
            
            if frame_index < total_frames:
                frame = vr[frame_index].asnumpy()
                
                # 直接将 RGB numpy 数组的原始字节写入 FFmpeg 的 stdin
                # 无需再用 cv2.cvtColor 进行颜色空间转换！
                try:
                    ffmpeg_process.stdin.write(frame.tobytes())
                except (IOError, BrokenPipeError) as e:
                    # 如果 FFmpeg 进程提前退出，管道会损坏，这里会捕获到异常
                    print(f"\n写入 FFmpeg 管道时出错: {e}")
                    print("FFmpeg 可能已因错误而终止。")
                    break

                print(f"\r正在处理: {i+1}/{num_seconds_to_sample} 秒...", end="")

        # 5. 关闭管道并等待 FFmpeg 完成处理
        print("\n所有帧已发送给 FFmpeg，等待编码完成...")
        # communicate() 会做这件事，并等待进程结束
        stderr_output = ffmpeg_process.communicate()[1]
        
        if ffmpeg_process.returncode == 0:
            print(f"FFmpeg 处理成功！已生成新视频: {sampled_video_path}")
            return sampled_video_path
        else:
            print(f"FFmpeg 处理失败！返回码: {ffmpeg_process.returncode}")
            print("--- FFmpeg 错误信息 ---")
            # 将字节串解码为字符串以便打印
            print(stderr_output.decode('utf-8'))
            print("-----------------------")
            return video_path # 出错则返回原视频路径

    except FileNotFoundError:
        print("\n错误: 'ffmpeg' 命令未找到。请确保 FFmpeg 已安装并配置在系统 PATH 环境变量中。")
        return video_path
    except Exception as e:
        print(f"\n视频预处理过程中发生未知错误: {e}")
        return video_path




if __name__=="__main__":
    #阿里百炼
    import os
    from dashscope import MultiModalConversation
    # 将xxxx/test.mp4替换为你本地视频的绝对路径
    local_path = "/Users/hanrui/Desktop/Haojing/pycharmproject/TransNetV2/output_scenes/scene_004_sampled.mp4"
    video_path = f"file://{local_path}"
    messages = [{'role': 'system',
                    'content': [
                        {'text': """你是一位专业的视频内容分析师。你的任务是忽略纯粹的视觉美学（如颜色、布局、背景），专注于视频中发生的核心事件、动作和意图。
                            请根据以下提供的视频信息，精准地概括出视频中包含的文字核心信息，直接输出内容摘要。在你的回答中，请不要使用‘视频展示了’、‘视频中’或类似的引导性词语。"""}
                        ]
                        },
                    {'role':'user',
                    # fps参数控制视频抽帧数量，表示每隔1/fps 秒抽取一帧
                    'content': [{'video': video_path,"fps":2},
                                {'text': '这段视频的主要内容是什么？请严格按照视频内容分析，不要有任何猜测信息，必须只用一句话'}
                                ]
                                }
                                ]
    response = MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key='sk-1968862eac2243d28563595b20f1f26b',
        model='qwen2.5-vl-72b-instruct',  
        messages=messages)
    print(response["output"]["choices"][0]["message"].content[0]["text"])









     # vedio_path="/Users/hanrui/Desktop/Haojing/pycharmproject/TransNetV2/output_scenes/scene_004.mp4"
    # pre_vedio_path= preprocess_video(vedio_path)
    #base64_video = encode_video('/Users/hanrui/Desktop/Haojing/pycharmproject/TransNetV2/output_scenes/scene_004_sampled.mp4')
   #LLM_API_KEY="ms-cb89ce66-03df-4308-b9c9-8cccded1d466"
    # LLM_SUMMARY_MODEL="Qwen/Qwen2.5-72B-Instruct-128K"
    # LLM_API_BASE_URL="https://lab.iwhalecloud.com/gpt-proxy/v1"
    # LLM_API_KEY="Bearer ailab_ifVKeBugo33NxWmVGt+fMy4aV1YREhcLs9VdD3c1I0QaRpjVbMykquPzeuKidmOwrCR9QSqlH8cGafOAxRyAwI3xxduBeRXNNSclSNnWV7DuWQ2G7y+jyyo="

    # client = OpenAI(
    # # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # api_key=LLM_API_KEY,
    # base_url=LLM_API_BASE_URL,
    # )
    # completion = client.chat.completions.create(
    # model=LLM_SUMMARY_MODEL,  
    # messages=[
    #     {
    #         "role": "system",
    #         "content": [{"type":"text","text": "你是个很有用的助手"
    # }]},
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "file:///Users/hanrui/Desktop/Haojing/pycharmproject/TransNetV2/output_scenes/scene_001.mp4",
    #                 "max_pixels": 360 * 420,
    #                 "fps": 1.0,
                    
    #             },
    #             {"type": "text", "text": "描述这个视频内容"},
    #         ],
    #     }
    # ],
    # )
    # print(completion.choices[0].message.content)