#read video and split into frames
import cv2
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import itertools

import argparse


def readvideo2PIL(video_path):
    video = cv2.VideoCapture(video_path)

    frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        converted = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(converted)
        frames.append(pil_im)

    video.release()
    return frames


def opticalsplit_frame_to_frames(video_path, total):
    video = cv2.VideoCapture(video_path)
    success, first_frame = video.read()
    if not success:
        raise RuntimeError(f"Cannot read video {video_path}")
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mags = []
    frames = [first_frame]

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(np.mean(magnitude))
        prev_gray = gray
    video.release()

    # Select key frames
    if len(frames) == 1:
        final_select = [0]
    else:
        mags = (mags - np.min(mags)) / (np.max(mags) - np.min(mags) + 1e-6)
        y3 = gaussian_filter1d(mags, 6)
        peaksy3, _ = find_peaks(y3, distance=30, prominence=0.3)
        interest = []

        if len(frames) < total:
            final_select = list(range(len(frames)))
        else:
            if total == 0:
                final_select = [len(frames) // 2]
            else:
                interest_range = list(range(len(frames) - 1))
                if len(peaksy3) > 0 and len(peaksy3) + 1 < total:
                    numframes = total // (len(peaksy3) + 1)
                    for p in peaksy3:
                        region = list(set(interest_range).intersection(
                            range(p - 15, p + 15, max(1, 30 // numframes))))
                        interest.append(region)
                        interest_range = list(set(interest_range) - set(region))
                    interest = list(itertools.chain.from_iterable(interest))
                    interest = np.array(interest).flatten()
                    if len(interest) < total:
                        step = max(1, len(interest_range) // max(1, (total - len(interest))))
                        noninterest = interest_range[::step]
                    else:
                        noninterest = interest_range[::60]
                    final_select = list(set(interest) | set(noninterest))
                    final_select.sort()
                else:
                    final_select = list(range(0, len(frames), max(1, len(frames) // total)))
    final_select = final_select[:total]

    # Gather selected frames
    selected_frames = [frames[i] for i in final_select]

    return selected_frames



def opticalsplit_frame(video_path, filename, des_dir, count, total):
    video = cv2.VideoCapture(video_path)
    success, first_frame = video.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) 
    mags = []
    frames = [first_frame]
    while(video.isOpened()):
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,  
                                       None, 
                                       0.5, 3, 15, 3, 5, 1.2, 0) 
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        mag = np.mean(magnitude)
        mags.append(mag)
        prev_gray = gray 
    if len(frames) == 1:
        final_select = [0]
        count[filename.split('.')[0]] = 0
    else:
        mags = (mags - np.min(mags)) / (np.max(mags) - np.min(mags))
        y3 = gaussian_filter1d(mags, 6)
        peaksy3, _ = find_peaks(y3,  distance=30, prominence=0.3)
        interest = []
        if len(frames) < total:
            final_select = list(range(0, len(frames)))
            count[filename.split('.')[0]] = 0
        else:
            if total == 0:
                final_select = [int(len(frames)/2)]
                count[filename.split('.')[0]] = 0
            else:
                interest_range = list(range(0, len(frames)-1))
                if len(peaksy3) > 0 and len(peaksy3)+ 1 < total:
                    count[filename.split('.')[0]] = len(peaksy3)
                    numframes = total // (len(peaksy3)+1)
                    for i in range(len(peaksy3)):
                        interest.append(list(set(interest_range).intersection(range(peaksy3[i]-15, peaksy3[i]+15,(30//numframes))) ))
                        interest_range = list(set(interest_range) - set(range(peaksy3[i]-15, peaksy3[i]+15)))
                    interest = list(itertools.chain.from_iterable(interest))
                    interest = np.array(interest).flatten()
                    if len(interest) < total:
                        noninterest = interest_range[::len(interest_range)// (total - len(interest))]
                    else:
                        noninterest = interest_range[::60]
                    final_select = list(set(interest) | set(noninterest))
                    final_select.sort()
                else:
                    count[filename.split('.')[0]] = 0
                    final_select = list(range(0, len(frames), len(frames)//total))
    final_select = final_select[:total]
    if not os.path.exists(os.path.join(des_dir,filename.split('.')[0])):
        os.makedirs(os.path.join(des_dir,filename.split('.')[0]))
    for i in final_select:
        cv2.imwrite(os.path.join(des_dir,filename.split('.')[0], "%s.jpg" % str(i).zfill(4)), frames[i])
    video.release()
    return count



def main(args):
    with open(args.input_json, 'r') as f:
        data = json.load(f)

    count = {}
    for item in tqdm(data):
        video_file = item[0]
        video_path = os.path.join(args.video_root, video_file)
        count = opticalsplit_frame(video_path, video_file, args.output_dir, count, args.total_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract selected frames from videos based on optical flow.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--video_root", type=str, required=True, help="Directory containing the video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted frames.")
    parser.add_argument("--total_frames", type=int, default=6, help="Number of frames to extract per video.")

    args = parser.parse_args()
    main(args)

