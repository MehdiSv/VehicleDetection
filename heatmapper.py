import collections

import numpy as np


class Heatmapper:
    def __init__(self):
        self.frame_boxes_list = collections.deque(maxlen=10)

    def add_frame_boxes(self, boxes):
        self.frame_boxes_list.append(boxes)

    def compute_heatmap(self, img):
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

        for frame_boxes in self.frame_boxes_list:
            for box in frame_boxes:
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return self.apply_threshold(heatmap, 10)

    @staticmethod
    def apply_threshold(heatmap, threshold):
        heatmap[heatmap <= threshold] = 0
        return heatmap
