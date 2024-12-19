import os
import numpy as np
import torch
import pickle
import logging
from torch.utils.data import Dataset, DataLoader


def unpack_fps(packed_fps):
    # packed_fps = np.array(packed_fps)
    shape = (*(packed_fps.shape[:-1]), -1)
    fps = np.unpackbits(packed_fps.reshape((-1, packed_fps.shape[-1])),
                        axis=-1)
    fps = torch.FloatTensor(fps).view(shape)

    return fps


class ValueDataset(Dataset):
    def __init__(self, fp_value_f):
        assert os.path.exists(fp_value_f)
        logging.info('Loading value dataset from %s' % fp_value_f)
        data_dict = torch.load('%s' % fp_value_f)
        self.fps = unpack_fps(data_dict['fps'])
        self.values = data_dict['values']

        filter = self.values[:, 0] > 0
        self.fps = self.fps[filter]
        self.values = self.values[filter]

        self.reaction_costs = data_dict['reaction_costs']
        self.target_values = data_dict['target_values']
        # self.reactant_fps = unpack_fps(data_dict['reactant_fps'])
        self.reactant_packed_fps = data_dict['reactant_fps']
        self.reactant_masks = data_dict['reactant_masks']
        self.reactant_fps = None
        self.reshuffle()

        assert self.fps.shape[0] == self.values.shape[0]
        logging.info('%d (fp, value) pairs loaded' % self.fps.shape[0])
        logging.info('%d nagative samples loaded' % self.reactant_fps.shape[0])
        print(self.fps.shape, self.values.shape,
              self.reactant_fps.shape, self.reactant_masks.shape)

        logging.info(
            'mean: %f, std:%f, min: %f, max: %f, zeros: %f' %
            (self.values.mean(), self.values.std(), self.values.min(),
             self.values.max(), (self.values == 0).sum() * 1. / self.fps.shape[0])
        )

    def reshuffle(self):
        shuffle_idx = np.random.permutation(self.reaction_costs.shape[0])
        self.reaction_costs = self.reaction_costs[shuffle_idx]
        self.target_values = self.target_values[shuffle_idx]
        self.reactant_packed_fps = self.reactant_packed_fps[shuffle_idx]
        self.reactant_masks = self.reactant_masks[shuffle_idx]

        self.reactant_fps = unpack_fps(
            self.reactant_packed_fps[:self.fps.shape[0], :, :])

    def __len__(self):
        return self.fps.shape[0]

    def __getitem__(self, index):
        return self.fps[index], self.values[index], \
            self.reaction_costs[index], self.target_values[index], \
            self.reactant_fps[index], self.reactant_masks[index]


class ValueDataLoader(DataLoader):
    def __init__(self, fp_value_f, batch_size, shuffle=True):
        self.dataset = ValueDataset(fp_value_f)

        super(ValueDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def reshuffle(self):
        self.dataset.reshuffle()


class MaxDepthValueDataset(Dataset):
    def __init__(self, fp_value_f, depth_filter=1):
        assert os.path.exists(fp_value_f)
        print('Loading MaxDepth dataset from %s' % fp_value_f)
        data_dict = torch.load('%s' % fp_value_f)
        self.fps = np.concatenate(data_dict['target_fps'], axis=0)
        self.fps = unpack_fps(self.fps)
        self.maxDepth = torch.tensor(data_dict['target_maxdepth']).view(-1, 1)
        assert self.fps.shape[0] == self.maxDepth.shape[0]
        filter = self.maxDepth[:, 0] >= depth_filter # 深度为0的分子没必要预测，本来就在可买数据集中
        self.fps = self.fps[filter]
        self.maxDepth = self.maxDepth[filter]
        self.label = self.maxDepth - depth_filter # 使得类别标签从0开始 label 0 == depth 1 ...
        print('%d (fp, depth) pairs loaded' % self.fps.shape[0])
        print(self.fps.shape, self.maxDepth.shape)

        self.all_depth = []
        for c in range(self.maxDepth.min().item(), self.maxDepth.max().item()+1):

            c_number = (self.maxDepth == c).sum()
            if c_number != 0:
                self.all_depth.append(c)
            print('Depth {}: {:.2f}%'.format(
                c, 100*c_number/self.maxDepth.shape[0]))
        self.reshuffle()
        

    def reshuffle(self):
        shuffle_idx = np.random.permutation(self.maxDepth.shape[0])
        self.fps = self.fps[shuffle_idx]
        self.maxDepth = self.maxDepth[shuffle_idx]
        self.label = self.label[shuffle_idx]

    def __len__(self):
        return self.fps.shape[0]

    def __getitem__(self, index):
        return self.fps[index], self.label[index]


class MaxDepthValueDatasetEmpty(Dataset):
    def __init__(self, fps, label):
        self.fps = fps
        self.label = label

    def reshuffle(self):
        shuffle_idx = np.random.permutation(self.label.shape[0])
        self.fps = self.fps[shuffle_idx]
        self.label = self.label[shuffle_idx]

    def __len__(self):
        return self.fps.shape[0]

    def __getitem__(self, index):
        return self.fps[index], self.label[index]

class MaxDepthValueDataLoader(DataLoader):
    def __init__(self, fp_value_f=None, dataset=None, batch_size=128, shuffle=True):
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = MaxDepthValueDataset(fp_value_f)

        super(MaxDepthValueDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def reshuffle(self):
        self.dataset.reshuffle()
