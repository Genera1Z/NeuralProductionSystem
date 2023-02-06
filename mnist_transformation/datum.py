import random

import PIL.Image as pili
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torchvision.datasets as ptvd
import torchvision.transforms as ptvt
import torchvision.transforms.functional as ptvtf


class MnistOperationSet(ptvd.MNIST):
    nearest = ptvtf.InterpolationMode.NEAREST
    operators = (
        lambda _: ptvtf.rotate(_, 60, MnistOperationSet.nearest),  # rotate_left
        lambda _: ptvtf.rotate(_, -60, MnistOperationSet.nearest),  # rotate_right
        lambda _: ptvtf.affine(_, 0, [0, -.15 * _.size(-2)], 1, [0, 0], MnistOperationSet.nearest),  # translate_up
        lambda _: ptvtf.affine(_, 0, [0, .15 * _.size(-2)], 1, [0, 0], MnistOperationSet.nearest)  # translate_down
    )

    def __init__(self, n_operation, root, train, transform: list):
        super(MnistOperationSet, self).__init__(root, train, download=True)
        self.n_operation = n_operation
        self.transform = ptvt.Compose(transform)
        self.data = self.data.numpy()  # (n,h,w)
        self.targets = self.targets.numpy()  # (n,)

    def __getitem__(self, idx):
        _image = pili.fromarray(self.data[idx], mode='L')  # TODO 为啥这俩行就可以实现a增加通道维度b归一化？
        image = self.transform(_image)  # (c,h,w)
        frame = ptvtf.pad(image, (64 - 28) // 2)
        frames = [frame]
        operations = []

        for t in range(self.n_operation):
            opid = random.choice(range(len(self.operators)))
            operations.append(opid)
            frame = self.operators[opid](frame)
            frames.append(frame)

        frames = pt.stack(frames, dim=0).float()
        operations = ptnf.one_hot(pt.from_numpy(np.array(operations, dtype='int64')), len(self.operators)).float()
        return frames, operations  # (t,c,h,w), (t,c)
