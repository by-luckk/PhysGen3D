import subprocess
import os

# 定义视频文件所在的目录
video_dir = 'videos'

# 遍历目录下的所有文件
for filename in os.listdir(video_dir):
    if filename.endswith('.mp4'):
        input_video = video_dir + '/' + filename
        output_video = video_dir + '2/' + filename
        print(input_video)

        command = [
            "ffmpeg",
            "-i", input_video,
            "-c:v", "libx264",  # 使用 H.264 编码器
            "-preset", "slow",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_video
        ]

        # 执行 FFmpeg 命令
        subprocess.run(command)