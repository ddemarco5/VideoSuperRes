import os
import sys
import time
import queue
import threading
import traceback

from enum import Enum
from queue import Queue
from threading import Lock, Thread

import numpy

import src.libav_functions as libav_functions

# for testing
import random

# an enum for now in case we expand commands in the future
class Command(Enum):
    STOP = 1
    ERROR = 2

class ThreadedDecoder:
    def __init__(self, path_name, max_ram_size):
        print("\n\nTHREADED DECODER INIT\n\n")
        self.path_name = path_name
        # build a list of all our video files
        self.video_files = [os.path.join(self.path_name, x) for x in os.listdir(self.path_name)]
        # The most amount of ram we're allowed to use
        self.max_ram_size = max_ram_size

        # Our two dicts we'll use to store the data
        self.buf_1 = []
        self.buf_1_lock = Lock()
        self.buf_2 = []
        self.buf_2_lock = Lock()
        # The active buffer we're feeding to the data loader
        self.active_buf = self.buf_1
        # The backup buffer
        self.backup_buf = self.buf_2
        # just use the first entry of the video file list to probe for video dims
        vid_width, video_height = libav_functions.get_video_dims(self.video_files[0])
        cache_entry_size = numpy.empty(shape=(vid_width,video_height,3), dtype=numpy.uint8).nbytes
        print("cache entry size:", cache_entry_size)
        # turn cache_size into bytes and divide by the byte size of each entry, and we need 2, so div by 2
        self.max_buf_size = int(((max_ram_size*1024*1024*1024)/cache_entry_size)//2)
        print("buf size:", self.max_buf_size)

        # build our chunk superlist we'll use to load data
        self.chunk_superlist = {}
        self.build_chunk_superlist()

        # fill our initial buffer with data
        self.fill_buffer(self.active_buf, block=True)
        # and lock it as active
        self.buf_1_lock.acquire()

        # keep running, but fill our second buffer in the background
        self.fill_buffer(self.backup_buf)

    # build a list of all chunks and their lengths to load into ram
    # TODO: not very robust, needs a lot more checking to deal with file types it might pull.
    def build_chunk_superlist(self):

        superlist = []

        # figure out how many frames are in each of our files
        files_sizes = []
        for f in self.video_files:
            files_sizes.append((f, libav_functions.get_total_frames(f)))

        # create offsets and lengths into the file based on our buffer size
        for filename, frames in files_sizes:
            #print("frames:", frames)
            chunk_size = self.max_buf_size
            # max number of full chunks
            full_chunks = frames//chunk_size
            #print("full chunks:", full_chunks)
            #print("chunk size:", chunk_size)
            start = 0
            for _ in range(0,full_chunks):
                listitem = (filename, start, chunk_size)
                superlist.append(listitem)
                start = start + chunk_size + 1
            # add our remainder straggler
            superlist.append((filename, start, frames - start))

        random.shuffle(superlist)
        #print(superlist)
        #print("we have {} unique chunks".format(len(superlist)))
        self.chunk_superlist = superlist

    def get_num_chunks(self):
        return len(self.chunk_superlist)

    # get the current active buffer's length
    def get_length(self):
        return len(self.active_buf)

    def worker_thread(self, path_name, start_frame, number_of_frames, target_buffer, buffer_lock):


        assert target_buffer is not None
        # Lock our buffer to make sure we're the only one touching it
        #print("Trying to grab lock:", buffer_lock)
        buffer_lock.acquire(timeout=10)

        # wipe the buffer, we'll be building it again
        target_buffer.clear()

        # decode the frames we want and send them directly to the buffer
        print("filling buffer")
        libav_functions.get_video_frames(path_name,
                                         start_frame=start_frame,
                                         number_of_frames=number_of_frames,
                                         target_buffer=target_buffer)
        print("buffer filled")

        # unlock our buffer and give it back
        buffer_lock.release()


    # swap the active buffer and refill the inactive
    def swap(self):
        if self.active_buf is self.buf_1:
            # release the lock and swap the buffer
            if self.buf_1_lock.locked():
                self.buf_1_lock.release()
            self.buf_2_lock.acquire(timeout=60)
            print("Swapping to buf 2!")
            self.active_buf = self.buf_2
            self.backup_buf = self.buf_1
            # Fill our inactive buffer with data
            self.fill_buffer(self.backup_buf)
        elif self.active_buf is self.buf_2:
            # release the lock and swap the buffer
            if self.buf_2_lock.locked():
                self.buf_2_lock.release()
            self.buf_1_lock.acquire(timeout=60)
            print("Swapping to buf 1!")
            self.active_buf = self.buf_1
            self.backup_buf = self.buf_2
            # Fill our inactive buffer with data
            self.fill_buffer(self.backup_buf)
        else:
            assert False, "BUG! Should never get here"

    # Pull a chunk off the superlist and load it into the backup buffer
    def fill_buffer(self, buffer, block=False):

        assert len(self.chunk_superlist) > 0, "No chunks in superlist!"

        # Grab a shuffled chunk
        filename, start_frame, length = self.chunk_superlist.pop()

        target_buffer = None
        if buffer is self.buf_1:
            target_buffer = self.buf_1
            buffer_lock = self.buf_1_lock
        elif buffer is self.buf_2:
            target_buffer = self.buf_2
            buffer_lock = self.buf_2_lock
        else:
            assert False, "BUG! Should never get here"

        print("loading {}, {}, {}".format(filename, start_frame, length))

        worker_thread = Thread(target=self.worker_thread, args=(filename,
                                                                start_frame,
                                                                length,
                                                                target_buffer,
                                                                buffer_lock,))
        # just fire the thread off and let it kill itself, don't worry about re-joining
        worker_thread.start()
        if block:
            worker_thread.join()
