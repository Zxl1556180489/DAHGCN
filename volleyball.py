import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

from PIL import Image
import random

import sys

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""
FRAMES_SIZE = {1: (720, 1280), 2: (720, 1280), 3: (720, 1280), 4: (720, 1280), 5: (720, 1280), 6: (720, 1280),
               7: (720, 1280), 8: (720, 1280), 9: (720, 1280), 10: (720, 1280),
               11: (720, 1280), 12: (720, 1280), 13: (720, 1280), 14: (720, 1280), 15: (720, 1280), 16: (720, 1280),
               17: (720, 1280), 18: (720, 1280), 19: (720, 1280), 20: (720, 1280),
               21: (720, 1280), 22: (720, 1280), 23: (720, 1280), 24: (720, 1280), 25: (720, 1280), 26: (720, 1280),
               27: (720, 1280), 28: (720, 1280), 29: (720, 1280), 30: (720, 1280),
               31: (720, 1280), 32: (720, 1280), 33: (720, 1280), 34: (720, 1280), 35: (720, 1280), 36: (720, 1280),
               37: (720, 1280), 38: (720, 1280), 39: (720, 1280), 40: (720, 1280),
               41: (720, 1280), 42: (720, 1280), 43: (720, 1280), 44: (720, 1280), 45: (720, 1280), 46: (720, 1280),
               47: (720, 1280), 48: (720, 1280), 49: (720, 1280), 50: (720, 1280),
               51: (720, 1280), 52: (720, 1280), 53: (720, 1280), 54: (720, 1280), 55: (720, 1280), }

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9


def volley_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    # 合并pass和set

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0]
            activity = gact_to_id[values[1]]

            values = values[2:]
            num_people = len(values) // 5

            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y + h, x + w

            bboxes = np.array([_read_bbox(values[i:i + 4])
                               for i in range(0, 5 * num_people, 5)])

            fid = int(file_name.split('.')[0])
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations


def volley_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
    return data


def volley_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames


def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid - num_before, src_fid + num_after + 1)]


def load_samples_sequence(anns, tracks, images_path, frames, image_size, num_boxes=12, ):
    """
    load samples of a bath

    Returns:
        pytorch tensors
    """
    images, boxes, boxes_idx = [], [], []
    activities, actions = [], []
    for i, (sid, src_fid, fid) in enumerate(frames):
        # img=skimage.io.imread(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        # img=skimage.transform.resize(img,(720, 1280),anti_aliasing=True)

        img = Image.open(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

        img = transforms.functional.resize(img, image_size)
        img = np.array(img)

        # H,W,3 -> 3,H,W
        img = img.transpose(2, 0, 1)
        images.append(img)

        boxes.append(tracks[(sid, src_fid)][fid])
        actions.append(anns[sid][src_fid]['actions'])
        if len(boxes[-1]) != num_boxes:
            boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes - len(boxes[-1])]])
            actions[-1] = actions[-1] + actions[-1][:num_boxes - len(actions[-1])]
        boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
        activities.append(anns[sid][src_fid]['group_activity'])

    images = np.stack(images)
    activities = np.array(activities, dtype=np.int32)
    bboxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
    bboxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
    actions = np.hstack(actions).reshape([-1, num_boxes])

    # convert to pytorch tensor
    images = torch.from_numpy(images).float()
    bboxes = torch.from_numpy(bboxes).float()
    bboxes_idx = torch.from_numpy(bboxes_idx).int()
    actions = torch.from_numpy(actions).long()
    activities = torch.from_numpy(activities).long()

    return images, bboxes, bboxes_idx, actions, activities


class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """

    def __init__(self, anns, tracks, frames, images_path, image_size, feature_size, num_boxes=12, num_before=4,
                 num_after=4, num_frames=3, is_training=True, is_finetune=False):
        self.anns = anns
        self.tracks = tracks
        self.frames = frames
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_frame = num_frames
        self.num_boxes = num_boxes
        self.num_before = num_before
        self.num_after = num_after

        self.is_training = is_training
        self.is_finetune = is_finetune

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)

    def __getitem__(self, index):  # index：3156
        """
        Generate one sample of the dataset
        """

        select_frames = self.volley_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)

        return sample

    def volley_frames_sample(self, frame):  # frame：（30 32780）

        sid, src_fid = frame  # sid：30  src_fid：32780

        if self.is_finetune:  # self.is_finetune = True for stage 1
            if self.is_training:  # self.is_training = True for stage 1  控制采样1帧还是10帧
                # fid=random.randint(src_fid-self.num_before, src_fid+self.num_after)  #在中心帧附近随机抽1帧
                # return [(sid, src_fid, fid)]
                sample_frames = random.sample(range(src_fid - self.num_before, src_fid + self.num_after + 1),
                                              self.num_frame)
                return [(sid, src_fid, fid)
                        for fid in sample_frames]
            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid - self.num_before, src_fid + self.num_after + 1)]  # 在中心帧附近随机抽10帧
        else:
            if self.is_training:
                sample_frames = random.sample(range(src_fid - self.num_before, src_fid + self.num_after + 1),
                                              self.num_frame)
                return [(sid, src_fid, fid)
                        for fid in sample_frames]

                # # START: Original code by ZXL
                # if src_fid == 1:
                #     sample_frames=random.sample(range(src_fid,src_fid+self.num_frames),10)
                #     return [(sid, src_fid, fid) for fid in sample_frames]
                # else:
                #     return [(sid, src_fid, fid)
                #         # for fid in[src_fid+5,src_fid - 3, src_fid, src_fid + 3, src_fid - 4, src_fid - 1, src_fid + 2, src_fid - 2,src_fid + 1, src_fid + 4]]
                #         for fid in
                #         [src_fid - 4, src_fid - 3, src_fid - 2, src_fid - 1, src_fid, src_fid + 1, src_fid + 2, src_fid + 3,
                #          src_fid + 4, src_fid + 5]]
                # # END: Original code by ZXL

            else:
                return [(sid, src_fid, fid)
                        for fid in
                        [src_fid - 3, src_fid, src_fid + 3, src_fid - 4, src_fid - 1, src_fid + 2, src_fid - 2,
                         src_fid + 1, src_fid + 4]]

    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """

        OH, OW = self.feature_size

        images, boxes = [], []
        activities, actions = [], []
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)
            images.append(img)

            temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][fid])
            for i, track in enumerate(self.tracks[(sid, src_fid)][fid]):
                y1, x1, y2, x2 = track
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                temp_boxes[i] = np.array([w1, h1, w2, h2])

            boxes.append(temp_boxes)

            actions.append(self.anns[sid][src_fid]['actions'])

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes - len(boxes[-1])]])
                actions[-1] = actions[-1] + actions[-1][:self.num_boxes - len(actions[-1])]
            activities.append(self.anns[sid][src_fid]['group_activity'])

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        actions = np.hstack(actions).reshape([-1, self.num_boxes])

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()

        return images, bboxes, actions, activities

