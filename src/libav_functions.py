<<<<<<< Updated upstream
import numpy

import ffmpeg
from math import floor
from fractions import Fraction

def get_video_dims(video_path):
    probe = ffmpeg.probe(video_path)
    container_info = probe['format']

    # get the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    return (int(video_stream['width']), int(video_stream['height']))

def get_total_frames(video_path):
    probe = ffmpeg.probe(video_path)
    container_info = probe['format']

    # get the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    video_stream_time_base = float(Fraction(video_stream['codec_time_base']))
    video_stream_framerate = float(Fraction(video_stream['avg_frame_rate']))

    # duration divided by av's time base
    container_duration_seconds = float(container_info['duration'])

    # Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)

    return num_frames

def get_video_frames(video_path, start_frame, number_of_frames, target_buffer):
    """
    Will fill target_buffer with number_of_frames starting at start_frame from video_path
    """

    # input checks
    #assert start_frame > 0, "Oops! Can't grab frame 0! Must start at 1"
    assert start_frame < get_total_frames(video_path), "Asked to get a frame out of range"

    probe = ffmpeg.probe(video_path)
    container_info = probe['format']

    # get the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    video_stream_time_base = float(Fraction(video_stream['codec_time_base']))
    video_stream_framerate = float(Fraction(video_stream['avg_frame_rate']))
    video_width = int(video_stream['width'])
    video_height = int(video_stream['height'])
    #print("video stream time base: ", video_stream_time_base)
    #print("video stream framerate: ", video_stream_framerate)

    # duration divided by av's time base
    container_duration_seconds = float(container_info['duration'])
    #print("duration (seconds): ", container_duration_seconds)

    # Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)
    #print("Calculated number of frames is: ", num_frames)

    # Calculate the time offset from the frame we want to seek
    desired_frame = start_frame
    seek_time_seconds = round(desired_frame / video_stream_framerate, 4)

    # TODO: find a way to shut this up. quiet=True breaks it for some reason
    ffmpeg_frame_grab_process = (
                    # we can use loglevel error here to quiet ffmpeg's stderr out
                    ffmpeg.input(video_path, ss=seek_time_seconds, loglevel="error")
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=number_of_frames)
                    .run_async(pipe_stdout=True)
                )

    #out, err = ffmpeg_frame_grab_command.run(quiet=True)

    frame_size = video_height * video_width * 3 # for rbg
    frames_got = 0
    while True:
        in_bytes = ffmpeg_frame_grab_process.stdout.read(frame_size)
        #print("read {} bytes".format(len(in_bytes)))
        if not in_bytes:
            #print("no bytes left!")
            break
        # the 3 here is because we're R G B
        frame_ndarray = numpy.frombuffer(in_bytes, numpy.uint8).reshape([video_height, video_width, 3])
        target_buffer.append(frame_ndarray)
        if not (frames_got % 100): # every 100 frames put out a little dot so we know we're working
            print(".", end="", flush=True)
        frames_got = frames_got + 1
    
    #print("read {} frames".format(num))

    ffmpeg_frame_grab_process.wait()

def get_video_frame(video_path, start_frame):
    """
    Takes an open video container and a target frame number, and will extract and return that frame
    in an rgb24 format ndarray
    """

    # input checks
    #assert start_frame > 0, "Oops! Can't grab frame 0! Must start at 1"
    assert start_frame < get_total_frames(video_path), "Asked to get a frame out of range"

    probe = ffmpeg.probe(video_path)
    container_info = probe['format']

    # get the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    video_stream_time_base = float(Fraction(video_stream['codec_time_base']))
    video_stream_framerate = float(Fraction(video_stream['avg_frame_rate']))
    video_width = int(video_stream['width'])
    video_height = int(video_stream['height'])
    #print("video stream time base: ", video_stream_time_base)
    #print("video stream framerate: ", video_stream_framerate)

    # duration divided by av's time base
    container_duration_seconds = float(container_info['duration'])
    #print("duration (seconds): ", container_duration_seconds)

    # Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)
    #print("Calculated number of frames is: ", num_frames)

    # Calculate the time offset from the frame we want to seek
    desired_frame = start_frame
    seek_time_seconds = round(desired_frame / video_stream_framerate, 4)

    ffmpeg_frame_grab_command = (
                    ffmpeg.input(video_path, ss=seek_time_seconds)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
                )


    out, err = ffmpeg_frame_grab_command.run(quiet=True)

    # the 3 here is because we're R G B
    frame_ndarray = numpy.frombuffer(out, numpy.uint8).reshape([video_height, video_width, 3])

    # need to return a copy because the version we get from frombuffer() is immutable
=======
import numpy

import ffmpeg
from math import floor, ceil
from fractions import Fraction

import subprocess

## takes a PIL image and compresses it with the chosen video coded and bitrate, and returns an ndarray
# bitrate is in Kbits/second
def compress_frame(pil_image, codec, bitrate):
    image_x, image_y = pil_image.size
    #numpy_image = pil_image.cpu().detach().numpy().astype(numpy.uint8)
    args = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(image_x, image_y) )
        .output("pipe:", format="mp4", pix_fmt="yuv420p", vcodec=codec, video_bitrate="{}K".format(bitrate))
        #.output("test.jpg")
        #.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        .compile()
    )
    #args.insert(len(args)-1, ["-movflags", "empty_moov"])
    args[11:11] = ["-movflags", "empty_moov"]
    #print(args)
    encode_process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    decode_process = (
        ffmpeg
        #.input("pipe:", format="mp4", pix_fmt="yuv420p", s="{}x{}".format(image_x, image_y) )
        .input("pipe:", format="mp4")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        #.output("test.jpg")
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    encode_process.stdin.write(pil_image.tobytes())
    encode_process.stdin.close()

    # read the bytes from our encode process
    encoded_bytes, eerr = encode_process.communicate(timeout=15)
    encode_process.wait()
    #print("Encoded {} bytes".format(len(encoded_bytes)))
    print(">", end='')

    # send the bytes to the decoding process to get the raw frame data out
    decode_process.stdin.write(encoded_bytes)
    #raw_frame = decode_process.stdout.read(image_x * image_y * 3)
    dout, derr = decode_process.communicate(input=encoded_bytes, timeout=15)

    #print(dout)
    #print(len(dout))
    #print("Decoded {} bytes".format(len(dout)))
    print("<", end='')

    encode_process.wait()
    decode_process.wait()


    #print(out)
    #print("\n")
    #print(err)

    processed_frame_nd = numpy.frombuffer(dout, numpy.uint8).reshape([image_x, image_y, 3])
    return processed_frame_nd
    #print(processed_frame_nd)


def get_video_dims(video_path):
    probe = ffmpeg.probe(video_path)
    container_info = probe['format']

    # get the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    return (int(video_stream['width']), int(video_stream['height']))

def get_total_frames(video_path):
    probe = ffmpeg.probe(video_path)
    container_info = probe['format']

    # get the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    video_stream_time_base = float(Fraction(video_stream['codec_time_base']))
    video_stream_framerate = float(Fraction(video_stream['avg_frame_rate']))
    
    #print(video_path)
    #print(video_stream_framerate)

    # duration divided by av's time base
    container_duration_seconds = float(container_info['duration'])
    
    #print(container_duration_seconds)

    # Calculate the number of frames this video file probably has
    #num_frames = floor(video_stream_framerate * container_duration_seconds)
    num_frames = ceil(video_stream_framerate * container_duration_seconds)
    #print(num_frames)


    return num_frames

def get_video_frames(video_path, start_frame, number_of_frames, target_buffer):
    """
    Will fill target_buffer with number_of_frames starting at start_frame from video_path
    """

    # input checks
    #assert start_frame > 0, "Oops! Can't grab frame 0! Must start at 1"
    assert start_frame < get_total_frames(video_path), "Asked to get a frame out of range"

    probe = ffmpeg.probe(video_path)
    container_info = probe['format']

    # get the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    video_stream_time_base = float(Fraction(video_stream['codec_time_base']))
    video_stream_framerate = float(Fraction(video_stream['avg_frame_rate']))
    video_width = int(video_stream['width'])
    video_height = int(video_stream['height'])
    #print("video stream time base: ", video_stream_time_base)
    #print("video stream framerate: ", video_stream_framerate)

    # duration divided by av's time base
    container_duration_seconds = float(container_info['duration'])
    #print("duration (seconds): ", container_duration_seconds)

    # Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)
    #print("Calculated number of frames is: ", num_frames)

    # Calculate the time offset from the frame we want to seek
    desired_frame = start_frame
    seek_time_seconds = round(desired_frame / video_stream_framerate, 4)

    # TODO: find a way to shut this up. quiet=True breaks it for some reason
    ffmpeg_frame_grab_process = (
                    # we can use loglevel error here to quiet ffmpeg's stderr out
                    ffmpeg.input(video_path, ss=seek_time_seconds, loglevel="error")
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=number_of_frames)
                    .run_async(pipe_stdout=True)
                )

    #out, err = ffmpeg_frame_grab_command.run(quiet=True)

    frame_size = video_height * video_width * 3 # for rbg
    frames_got = 0
    while True:
        in_bytes = ffmpeg_frame_grab_process.stdout.read(frame_size)
        #print("read {} bytes".format(len(in_bytes)))
        if not in_bytes:
            #print("no bytes left!")
            break
        # the 3 here is because we're R G B
        frame_ndarray = numpy.frombuffer(in_bytes, numpy.uint8).reshape([video_height, video_width, 3])
        target_buffer.append(frame_ndarray)
        if not (frames_got % 100): # every 100 frames put out a little dot so we know we're working
            print(".", end="", flush=True)
        frames_got = frames_got + 1
    
    #print("read {} frames".format(num))

    ffmpeg_frame_grab_process.wait()

def get_video_frame(video_path, start_frame):
    """
    Takes an open video container and a target frame number, and will extract and return that frame
    in an rgb24 format ndarray
    """

    # input checks
    #assert start_frame > 0, "Oops! Can't grab frame 0! Must start at 1"
    assert start_frame < get_total_frames(video_path), "Asked to get a frame out of range"

    probe = ffmpeg.probe(video_path)
    container_info = probe['format']

    # get the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    video_stream_time_base = float(Fraction(video_stream['codec_time_base']))
    video_stream_framerate = float(Fraction(video_stream['avg_frame_rate']))
    video_width = int(video_stream['width'])
    video_height = int(video_stream['height'])
    #print("video stream time base: ", video_stream_time_base)
    #print("video stream framerate: ", video_stream_framerate)

    # duration divided by av's time base
    container_duration_seconds = float(container_info['duration'])
    #print("duration (seconds): ", container_duration_seconds)

    # Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)
    #print("Calculated number of frames is: ", num_frames)

    # Calculate the time offset from the frame we want to seek
    desired_frame = start_frame
    seek_time_seconds = round(desired_frame / video_stream_framerate, 4)

    ffmpeg_frame_grab_command = (
                    ffmpeg.input(video_path, ss=seek_time_seconds)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
                )


    out, err = ffmpeg_frame_grab_command.run(quiet=True)

    # the 3 here is because we're R G B
    frame_ndarray = numpy.frombuffer(out, numpy.uint8).reshape([video_height, video_width, 3])

    # need to return a copy because the version we get from frombuffer() is immutable
>>>>>>> Stashed changes
    return frame_ndarray.copy()