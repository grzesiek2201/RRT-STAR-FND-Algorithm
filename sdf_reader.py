import xml.etree.ElementTree as ET
from typing import Optional, List, Tuple, Union
import numpy as np
from numpy import ndarray


class SDFReader:
    def __init__(self):
        self.tree = None
        self.root = None

    def parse(self, filepath: str) -> None:
        """ Parse file to tree """
        if filepath is None:
            return filepath
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()

    def get_obstacles(self):
        if self.tree is None:
            return None
        data = self.root.findall(".//visual/*[cylinder]/..")
        poses = np.array([np.array(list(map(float, child[0].text.split(" ")))) for child in data], dtype=object)
        radia = np.array([float(child[1][0][0].text) for child in data], dtype=object)
        pose_radia = list(zip(poses, radia))
        obstacles = np.array([[np.array([obstacle[0][0], obstacle[0][1]]), obstacle[1]] for obstacle in pose_radia], dtype=object)

        x = poses[:, 0] * 100
        y = poses[:, 1] * 100
        x_range = [np.min(x) - 100, np.max(x) + 100]
        y_range = [np.min(y) - 100, np.max(y) + 100]

        return obstacles, (x_range, y_range)
