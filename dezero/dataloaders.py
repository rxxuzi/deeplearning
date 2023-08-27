import numpy as np
import random
import math

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        """
        :param dataset: Datasetのインターフェースを持つインスタンス
        :param batch_size: バッチサイズ
        :param shuffle: エポック毎にデータセットをリセットするか
        :param gpu:
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        self.iteration += 1
        return batch_x, batch_t

    def next(self):
        return self.__next__()