import os, imageio
import skvideo
path="/home/ws/SoftwareSetup/miniconda3/envs/VQA/bin/"
skvideo.setFFmpegPath(path)
import skvideo.io

import cv2 as cv

# 获取gif图的最后一帧,并保存png
# def get_gif_last_frame(gif_path, image_path):
#     # image_path = gif_path.replace(".gif", ".png")
#     if os.path.exists(image_path):
#         return
#     else:
#         # try:
#         #     os.remove(image_path)
#         # except Exception as e:
#         #     print("Error removing file:", e)
#         with imageio.get_reader(gif_path) as reader:
#             last_frame = reader.get_data(reader.get_length() - 1)
#             imageio.imwrite(image_path, last_frame)
#         # return image_path

# 像素发生变化   舍弃
# def get_gif_last_frame(gif_path, image_path):
#     # image_path = gif_path.replace(".gif", ".png")
#     if os.path.exists(image_path):
#         return
#     else:
#         try:
#             os.remove(image_path)
#         except Exception as e:
#             print("Error removing file:", e)
#
#         try:
#             video_data = skvideo.io.vread(gif_path)
#             last_frame = video_data[-1]
#             cv.imwrite(image_path, last_frame)
#         except Exception as ex:
#             print(ex)

def get_gif_last_frame(gif_path, image_path):
    import ffmpeg

    metadata = ffmpeg.probe(gif_path)
    frame_num = int(metadata['streams'][0]['nb_frames']) - 1

    out, _ = (
        ffmpeg
            .input(gif_path)
            .filter('select', 'gte(n,{})'.format(frame_num))
            .output(image_path, vframes=1, loglevel=0)
            .run(overwrite_output=True)
    )