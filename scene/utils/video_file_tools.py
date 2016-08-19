import numpy as np
import cv2
import datetime
import random
import os
import math
import fnmatch


__author__ = 'xhou'


def shrink_img(img_mat, max_size):
    img_h, img_w = img_mat.shape[:2]
    ratio = min(float(max_size) / max(img_h, img_w), 1)
    new_h = int(img_h * ratio)
    new_w = int(img_w * ratio)
    return cv2.resize(img_mat, (new_w, new_h))


def get_video_len(in_video):
    cap = cv2.VideoCapture(in_video)
    # return int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# def extract_frames(in_video, frame_list, max_img_size=None):
#     # TODO: Replace cv2 reader with ffmpeg
#     try:
#         cap = cv2.VideoCapture(in_video)
#     except IOError:
#         print 'Cannot read file: {}'.format(in_video)
#         return
#
#     # video_len = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#     video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_list = np.array(frame_list)
#     frame_list.sort()
#
#     min_frame_ind = np.where(frame_list >= 0)[0][0]
#     max_frame_ind = np.where(frame_list < video_len)[0][-1]
#     frame_list = frame_list[min_frame_ind:max_frame_ind+1]
#
#     out_frames = []
#     for cur_frame_ind in range(video_len):
#         _, cur_frame = cap.read()
#         if cur_frame_ind in frame_list:
#             if max_img_size:
#                 cur_frame = shrink_img(cur_frame, max_img_size)
#             out_frames.append(cur_frame)
#
#         # if cur_frame_ind == frame_list[-1]:
#         #     break
#     return out_frames


# def extract_frames(in_video, frame_list, out_root):
#     piper = pipe.Pipeline()
#     save_key = in_video.split('/')[-1].split('.')[0]
#     for frame in frame_list:
#         piper.get_frame(in_video, os.path.join(out_root, save_key + '-' + str(frame) + '.jpg'), frame)



def _strCmp(s):
    return int(s.split('/')[-1].split('_')[0][1:])

def extract_frames(in_video, space, max_img_size=None, flag='video'):
    if flag == 'video':
        try:
            cap = cv2.VideoCapture(in_video)
        except IOError:
            print 'Cannot read file: {}'.format(in_video)
            return

        video_len = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frame_list = np.arange(0, video_len, space)

        out_frames = []
        for cur_frame_ind in range(video_len):
            _, cur_frame = cap.read()
            if cur_frame_ind in frame_list:
                if max_img_size:
                    cur_frame = shrink_img(cur_frame, max_img_size)
                out_frames.append(cur_frame)

    elif flag == 'images':
        frames = []
        for dir_name, _, file_list in os.walk(in_video):
            for name in file_list:
                if fnmatch.fnmatch(name, '*cam1*.jpg'):
                    print name
                    frames.append(os.path.join(dir_name, name))
        frames.sort(key=_strCmp)

        video_len = len(frames)
        frame_list = np.arange(0, video_len, space)

        out_frames = []
        for cur_frame_ind in range(video_len):
            if cur_frame_ind in frame_list:
                out_frames.append(cv2.imread(frames[cur_frame_ind]))
    else:
        raise ValueError

    return out_frames


def extract_frame_random(video_data_dir, video_list_file, save=False, frames_dir="", max_img_size=None):
    try:
        video_list = [l[:-1] for l in open(video_list_file)]
    except:
        print 'Cannot read file: {}'.format(video_list_file)
        return

    video_sample = video_list[random.randint(0, len(video_list) - 1)]
    try:
        cap = cv2.VideoCapture(os.path.join(video_data_dir, video_sample))
    except:
        print 'Cannot read file: {}'.format(video_sample)
        return
    video_sample_length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frame_sample_num = random.randint(1, video_sample_length)
    #print video_sample + '\t' + str(video_sample_length) + '\t' + str(frame_sample_num)
    itr = 0
    while itr < frame_sample_num:
        ret, frame_sample = cap.read()
        itr += 1
        if max_img_size:
            frame_sample = shrink_img(frame_sample, max_img_size)
    file_name = video_sample.split('/')[-1].split('.')[0] + '-' + str(frame_sample_num) + '.png'
    if save:
        cv2.imwrite(os.path.join(frames_dir, file_name), frame_sample)
    return frame_sample, video_sample


def extract_frame_set_random(sample_num, video_data_dir, video_list_file, max_img_size=None):
    out_frames = []
    for i in range(sample_num):
        frame_sample, pt = extract_frame_random(video_data_dir, video_list_file, max_img_size)
        print pt
        out_frames.append(frame_sample)
    return out_frames


def extract_frame_set_random_fast(dumpfolder, sample_num, video_data_dir, video_list_file, max_img_size=None):
    try:
        video_list = [l[:-1] for l in open(video_list_file)]
    except:
        print 'Cannot read file: {}'.format(video_list_file)
        return

    sample_num_per_frame = int(math.ceil(float(sample_num) / len(video_list)))
    frames = []
    frames_file_name = []
    print len(video_list)
    for vid in video_list:
        try:
            cap = cv2.VideoCapture(os.path.join(video_data_dir, vid))
        except:
            print 'Cannot read file: {}'.format(vid)
            return

        video_sample_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print int(math.ceil(sample_num / len(video_list)))
        frame_sample_list = random.sample(range(video_sample_length), sample_num_per_frame)
        print frame_sample_list, len(frame_sample_list), sample_num_per_frame

        itr = 0
        max_idx = max(frame_sample_list)
        while itr < max_idx:
            ret, frame_sample = cap.read()
            itr += 1
            if itr in frame_sample_list:
                print vid + '\t' + str(video_sample_length) + '\t' + str(itr)
                if max_img_size:
                    frame_sample = shrink_img(frame_sample, max_img_size)
                frames.append(frame_sample)
                frames_file_name.append(vid)

                frame_file_name = vid.replace('/', '~')
                prefix = frame_file_name.split('.')[0]
                # print dumpfolder + prefix + '_' + str(itr) + '.jpg'
                cv2.imwrite(dumpfolder + prefix + '_' + str(itr) + '.jpg', frame_sample)
    return frames, frames_file_name


def extract_random_frame_label_fast(p):
    dumpfolder = p.get('file', 'dumpfolder')
    video_data_dir = p.get('file', 'folder')
    video_list_file = p.get('file', 'filelist')
    # max_img_size = p.getint('gist_sample', 'max_img_size')
    sample_num = p.getint('gist_sample', 'sample_num')

    try:
        os.mkdir(dumpfolder)
    except:
        pass

    extract_frame_set_random_fast(dumpfolder, sample_num, video_data_dir, video_list_file, None)
    # ran_frm, frames_file_name = extract_frame_set_random_fast(dumpfolder, sample_num, video_data_dir, video_list_file, None)
    # for i in range(len(ran_frm)):
    #     frame_file_name = frames_file_name[i].replace('/', '-')
    #     prefix = frame_file_name.split('.')[0]
    #     print dumpfolder + prefix + '_' + str(i) + '.jpg'
    #     cv2.imwrite(dumpfolder + prefix + '_' + str(i) + '.jpg', ran_frm)

# TODO: function rename
# TODO: Should not use parameters: extract_random_frames(src_dir='', dst_dir='', sample_num=1000)
def extract_random_frame_label(p):
    dumpfolder = p.get('file', 'dumpfolder')
    video_data_dir = p.get('file', 'folder')
    video_list_file = p.get('file', 'filelist')
    max_img_size = p.getint('gist_sample', 'max_img_size')
    sample_num = p.getint('gist_sample', 'sample_num')

    for i in range(sample_num):
        ran_frm, sample_file_name = extract_frame_random(video_data_dir, video_list_file, None)
        sample_file_name = sample_file_name.replace('/', '-')
        prefix = sample_file_name.split('.')[0]
        try:
            os.mkdir(dumpfolder)
        # TODO: TOO BROAD!
        except:
            pass
        # print dumpfolder + prefix + '_' + str(i) + '.jpg'
        cv2.imwrite(dumpfolder + prefix + '_' + str(i) + '.jpg', ran_frm)


# TODO: Not necessary
def sec_to_min(sec):
    return str(datetime.timedelta(seconds=sec))


def extract_clip(in_video, clip_length, max_img_size=None, save=False, clips_dir=''):
    try:
        cap = cv2.VideoCapture(in_video)
    except IOError:
        print 'Cannot read file: {}'.format(in_video)
        return

    video_len = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    start = random.randint(1, video_len - clip_length - 1)
    itr = 0

    out_frames = []

    while itr < video_len and itr < start + clip_length:
        ret, cur_frame = cap.read()
        itr += 1
        if itr >= start:
            if max_img_size:
                cur_frame = shrink_img(cur_frame, max_img_size)
            out_frames.append(cur_frame)

    filename = in_video.split('/')[-1].split('.')[0] + "-" + str(start)
    if save and not(os.path.isfile(os.path.join(clips_dir, filename)+'.npz')):
        np.savez(os.path.join(clips_dir, filename), clip=np.array(out_frames))
        filename = os.path.join(clips_dir, filename) + '.npz'
    else:
        filename = None
    return out_frames, filename


def video_BGR_to_RGB(clip):
    result = []
    for frame in clip:
        result.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.array(result)


def show_video(clip):
    for frame in clip:
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyWindow('frame')


def video_meta_data(in_video, fps_default=15):
    try:
        cap = cv2.VideoCapture(in_video)
    except IOError:
        print 'Cannot read file: {}'.format(in_video)
        return

    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if np.isnan(cap.get(cv2.cv.CV_CAP_PROP_FPS)):
        fps = fps_default
    else:
        fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    # TODO: make a dictionary
    return width, height, length, fps