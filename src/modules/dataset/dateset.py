from __future__ import annotations
import numpy
import torch
from typing import Tuple
from dataclasses import dataclass
from .dateseries import DatasetDateSeries


# Tsr = torch.DoubleTensor
Tsr = torch.Tensor


def get_order(ti: Tsr) -> numpy.int32:
    m = ti.mean().item()
    n_digits = numpy.floor(numpy.log10(m)).astype(numpy.int32)
    o = 10 ** (n_digits//2)
    return o


def make_curve_cyclic(ti: Tsr) -> Tsr:
    o = get_order(ti)
    # return torch.sin(25/o * ti) + 3 * torch.sin(5/o * ti) + 0.5 * torch.cos(1/o * ti)
    return torch.sin(2/o * ti)


def make_curve_trend(ti: Tsr) -> Tsr:
    o = get_order(ti)
    return (1/o * ti ** 2 + 5/o * ti + 2)


@dataclass
class DateTensors(object):
    ti: Tsr = Tsr([])   # time index
    tv: Tsr = None      # time variables
    tc: Tsr = Tsr([])   # time constant
    kn: Tsr = Tsr([])
    tg: Tsr = Tsr([])
    device: torch.device = torch.device("cpu")

    def to(self) -> DateTensors:
        tensors = [self.ti, self.tv, self.tc, self.kn, self.tg]
        for tsr in tensors:
            if tsr is not None:
                tsr.to(self.device)

class DatesetToy(object):
    dct_curve: dict = dict(cyclic=make_curve_cyclic, trend=make_curve_trend)

    def __init__(self, Dout: int, wsz: int=7, model_type: str="cyclic", device: torch.device=torch.device("cpu")):
        super().__init__()
        self.Dout = Dout
        self.wsz = wsz
        self.model_type = model_type
        self.device = device
        self.date_series = None

        self.trainset = None
        self.testset = None

    def create_trainset(
        self,
        s="2016-01-01",
        e="2018-12-31",
        freq="D",
    ) -> DateTensors:
        """
        ti.shape = (N, W, Dout)
        """
        self.date_series = DatasetDateSeries(start=s, end=e, wsz=self.wsz, to_onehot=True)    # wsz same as W

        # ti window data to tensor
        ti = Tsr(self.date_series.ti_win)

        # tc window data to tensor
        N, W, Dtc = len(ti), self.date_series.wsz, 3
        tc = torch.randint(0, 2, (1, 1, Dtc)).repeat(N, W, 1)  # shape: (N, W, Dtc)

        # kn window data to tensor
        kn = Tsr(self.date_series.kn_win)

        # create target data as `tg` (target)
        tg = self.dct_curve[self.model_type](ti).repeat(1, 1, self.Dout)

        ti, tc, kn, tg = self.to_device(ti, tc, kn, tg)
        trainset = DateTensors(ti=ti, tc=tc, kn=kn, tg=tg, device=self.device)       # ti/tc/kn.shape: (N, W, Dout), tg.shape = (N, 1, Dout)
        self.trainset = trainset
        return trainset

    def to_device(self, ti: Tsr, tc: Tsr, kn: Tsr, tg: Tsr) -> Tuple[Tsr, Tsr, Tsr, Tsr]:
        device = self.device
        return ti.to(device), tc.to(device), kn.to(device), tg.to(device)

    def create_testset(
        self, by="2019-12-31"
    ) -> DateTensors:
        nxt = self.date_series.next_date
        test_series = DatasetDateSeries(
            start=nxt, end=by, wsz=self.date_series.wsz, wshift=self.date_series.wshift, to_onehot=True
        )
        # ti window data to tensor
        ti = Tsr(test_series.ti_win)

        # tc window data to tensor
        N, W = len(ti), test_series.wsz
        tc = (
            self.trainset.tc[0, 0, :].reshape(1, 1, -1).repeat(N, W, 1)
        )  # shape: (N, W, Dtc)

        kn = Tsr(test_series.kn_win)

        # create target data as `tg`(target)
        tg = self.dct_curve[self.model_type](ti).repeat(1, 1, self.Dout)

        ti, tc, kn, tg = self.to_device(ti, tc, kn, tg)
        testset = DateTensors(ti=ti, tc=tc, kn=kn, tg=tg, device=self.device)       # ti/tc/kn.shape: (N, W, Dout), tg.shape = (N, 1, Dout)
        self.testset = testset
        return testset


if __name__ == "__main__":
    # create toy dataset
    W, Dout = 14, 1
    toydataset = DatesetToy(Dout, W, "cycle", device=torch.device("cuda:0"))