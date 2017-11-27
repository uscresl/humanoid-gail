import numpy as np


class AmcParser(object):
    def __init__(self):
        self.frames = []

    def parse(self, file_name):
        amc_file = open(file_name, 'r')
        self.frames = []  # list of dicts {property: value(s)}
        current_frame = {}
        for i, line in enumerate(amc_file):
            if i < 3:
                continue
            if line.startswith('root'):
                if len(current_frame) > 0:
                    self.frames.append(current_frame)
                current_frame = {}
            split = line.split(' ')
            if len(split) <= 1:
                continue
            current_frame[split[0]] = np.array([float(x) for x in split[1:]])
        if len(current_frame) > 0:
            self.frames.append(current_frame)
