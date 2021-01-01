import numpy
import pandas
import torch
from typing import Union


BatchType = Union[
    numpy.ndarray,
    pandas.DataFrame,
    pandas.Series,
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
]


class BatchMaker(object):
    def __init__(self, bsz=64) -> None:
        self.bsz = bsz

    def __call__(self, data: BatchType) -> BatchType:
        n = len(data)
        for idx in range(0, n, self.bsz):
            bch = data[idx : idx + self.bsz]
            yield bch


if __name__ == "__main__":
    d = numpy.array(range(120)).reshape(-1, 3)
    batch = BatchMaker(bsz=4)
    for bch in batch(d):
        print(bch[:, 0])

    for bx, by, bz in zip(batch(d), batch(d), batch(d)):
        print(
            bx[0, 0],
            by[0, 0],
            bz[0, 0],
        )

    # test for tensor
    ti = tv = tc = kn = torch.arange(0, 60).view(4, 3, -1)
    print("tv.shape:", tv.shape)
    batch = BatchMaker(bsz=2)
    for bch in zip(batch(ti), batch(tv)):
        print(bch)
