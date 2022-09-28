import ffmpeg
import numpy as np

import torch

from torchvision.utils import save_image
import torchvision.transforms as transforms

import libav_functions

#out, _ = (
#    ffmpeg
#    .input('data/train/vid1.mkv', ss="00:01:00")
#    #.filter('select', 'gte(n,{})'.format(100))
#    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
#    .run(capture_stdout=True)
#)
ffmpeg_command = (
                    ffmpeg.input('data/train/vid1.mkv', ss="00:01:00")
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
                )
#frame_ndarray = np.frombuffer(out, np.uint8).reshape([1080, 1440, 3])

#for i in range(0,5):
#    print("Grabbing frame", i)
#    out, _ = ffmpeg_command.run(quiet=True)
#    frame_ndarray = np.frombuffer(out, np.uint8).reshape([1080, 1440, 3])
#    frame_tensor = transforms.ToTensor()(frame_ndarray.copy())
#    save_image(frame_tensor, "test/newlibtest.png")

libav_functions.get_total_frames('data/train/vid1.mkv')
libav_functions.get_video_frame('data/train/vid1.mkv', 100)