import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import src.networks as networks
import src.libav_functions

from skimage.exposure import match_histograms
import numpy

import random

checkpoint = torch.load("data/models/model.pth")
resblocks = networks.RRDB_Resblocks(8)
model = networks.Generator(resblocks)
model.load_state_dict(checkpoint['gen_model_state'])
model.eval()


video_file = "C:/Users/Dominic/Desktop/temp_torrents/ghost_stories_dvd_rip/Disc 1/title_t01.mkv"
#video_file = "C:/Users/Dominic/Desktop/temp_torrents/[Netaro] Dr. Stone S2 (BD 1080p HEVC FLAC) [Dual-Audio]/stone.mkv"
#video_file = "C:/Users/Dominic/Desktop/temp_torrents/[ILA] Jungle Tatei - Complete/[ILA] Jungle Tatei - 01 [6C527555].mkv"
#video_file = "C:/Users/Dominic/Desktop/temp_torrents/Ghost Stories (Gakkou no Kaidan) (2000) [kuchikirukia]/01. Ghost Stories (2000) [DVD 480p Hi10P AC3 dual-audio][kuchikirukia].mkv"
total_frames = src.libav_functions.get_total_frames(video_file)
print("total video frames:", total_frames)
desired_frame = random.randint(0,total_frames)
print("Trying upscaling on frame number", desired_frame)
frame_numpy = src.libav_functions.get_video_frame(
        video_file,
        desired_frame
    )
frame_tensor_raw = transforms.ToTensor()(frame_numpy)
original_frame_dims = list(frame_tensor_raw.size())[-2:]
frame_tensor = frame_tensor_raw.unsqueeze(0)

result = model(frame_tensor)
result_dims = list(result.size())[-2:]
print("orig dims:", original_frame_dims)
print("result dims:", result_dims)

orig_resized = transforms.Resize(result_dims)(frame_tensor)
pad_x = result_dims[0] - original_frame_dims[0]
pad_y = result_dims[1] - original_frame_dims[1]
#orig_resized = transforms.Pad(padding=(pad_y,pad_x,0,0))(frame_tensor)


# try using skimage to histogram normalize our output
#orig_nd = frame_tensor_raw.detach().numpy()
orig_nd = orig_resized.detach().numpy()
new_nd = result.detach().numpy()
print(new_nd.shape, orig_nd.shape)

#matched_new_nd = match_histograms(new_nd, orig_nd, multichannel=True)

#matched_new_nd_tensor = torch.from_numpy(matched_new_nd)[:1]

# back to tensor
#canvas = torch.cat([orig_resized[:1], result[:1], matched_new_nd_tensor], axis=3)
canvas = torch.cat([orig_resized[:1], result[:1]], axis=3)
save_image(canvas, "test/test_output.png")


#canvas = torch.cat([orig_nd_tensor, matched_new_nd_tensor], axis=3)
#save_image(canvas, "test/test_output_matched.png")