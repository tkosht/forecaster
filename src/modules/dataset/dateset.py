from __future__ import annotations
from typing import Tuple
import numpy
import pandas
import torch
from dataclasses import dataclass

from .batcher import BatchMaker, BatchType
from .dateseries import DatasetDateSeries


Tsr = torch.DoubleTensor
# Tsr = torch.Tensor


def get_basedate() -> numpy.int64:
    ti_base = pandas.date_range("2010-1-1", "2010-1-2")[0]
    ti_base = ti_base.to_numpy().astype(numpy.int64)
    return ti_base


def get_order(ti: Tsr) -> numpy.int32:
    ti_base = get_basedate()
    n_digits = numpy.int(numpy.floor(numpy.log10(ti_base)))
    o = 10 ** n_digits
    return o


def make_curve_cyclic(ti: Tsr) -> Tsr:
    b = get_basedate()
    _ti = ti - b
    o = get_order(_ti)
    _ti = _ti / (o // 1000) % numpy.pi  # _ti scales to (0, pi)
    curve = torch.sin(25 * _ti) + 3 * torch.sin(5 * _ti) + 0.5 * torch.cos(1 * _ti)
    return curve


def make_curve_trend(ti: Tsr) -> Tsr:
    b = get_basedate()
    _ti = ti - b
    o = get_order(_ti)
    _ti = _ti / o  # _ti scales to (0, 1)
    curve = 7 * (2 * _ti) ** 2 + 1000 * _ti + 2
    curve += 3 * make_curve_cyclic(ti)
    return curve


@dataclass
class DateTensors(object):
    ti: Tsr = Tsr([])  # time index, never none
    tv: Tsr = None  # time variables
    tc: Tsr = None  # time constant
    kn: Tsr = None  # known data even in future (like weekdays, ...)
    tg: Tsr = None  # target
    device: torch.device = torch.device("cpu")

    def to(self) -> DateTensors:
        tensors = [self.ti, self.tv, self.tc, self.kn, self.tg]
        for tsr in tensors:
            if tsr is not None:
                tsr.to(self.device)
        return self

    def create_shuffle(self) -> DateTensors:
        shuf = numpy.random.permutation(range(len(self)))
        ti, tv, tc, kn, tg = (
            self.ti[shuf],
            self.tv[shuf] if self.tv is not None else None,
            self.tc[shuf] if self.tc is not None else None,
            self.kn[shuf] if self.kn is not None else None,
            self.tg[shuf] if self.tg is not None else None,
        )
        return DateTensors(ti, tv, tc, kn, tg)

    def safe_tuple(self) -> Tuple[BatchType]:
        def _is_emp(t: Tsr):
            return (t is None) or (len(t[0]) == 0)

        empty = Tsr([]).to(self.ti.device)
        ti = self.ti
        tv = self.tv if not _is_emp(self.tv) else empty
        tc = self.tc if not _is_emp(self.tc) else empty
        kn = self.kn if not _is_emp(self.kn) else empty
        tg = self.tg if not _is_emp(self.tg) else empty
        return ti, tv, tc, kn, tg

    def __call__(self, bsz) -> BatchType:
        batch = BatchMaker(bsz)
        empty = [Tsr([])] * len(self.ti)
        ti = self.ti
        tv = self.tv if self.tv is not None else empty
        tc = self.tc if self.tc is not None else empty
        kn = self.kn if self.kn is not None else empty
        tg = self.tg if self.tg is not None else empty

        for bch in zip(batch(ti), batch(tv), batch(tc), batch(kn), batch(tg)):
            yield DateTensors(*bch, self.device)

    def __len__(self):
        return len(self.ti)

    def __eq__(self, other: DateTensors):
        is_equal = True

        def eq(tsr1: BatchType, tsr2: BatchType):
            if (tsr1 is not None) and (tsr2 is not None):
                return (tsr1 == tsr2).all()
            if (tsr1 is None) and (tsr2 is None):
                return True
            return False

        is_equal = is_equal and eq(self.ti, other.ti)
        is_equal = is_equal and eq(self.tv, other.tv)
        is_equal = is_equal and eq(self.tc, other.tc)
        is_equal = is_equal and eq(self.kn, other.kn)
        is_equal = is_equal and eq(self.tg, other.tg)
        return is_equal


class DatesetToy(object):
    dct_curve: dict = dict(cyclic=make_curve_cyclic, trend=make_curve_trend)

    def __init__(
        self,
        Dout: int,
        wsz: int = 7,
        model_type: str = "cyclic",
        device: torch.device = torch.device("cpu"),
    ):
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
        self.date_series = DatasetDateSeries(
            start=s, end=e, wsz=self.wsz, to_onehot=True
        )  # wsz same as W

        # ti window data to tensor
        ti = Tsr(self.date_series.ti_win)

        # tc window data to tensor
        # # N, W, Dtc = len(ti), self.date_series.wsz, 3
        # # tc = torch.randint(0, 2, (1, 1, Dtc)).repeat(N, W, 1)  # shape: (N, W, Dtc)
        N, W = len(ti), self.date_series.wsz
        tc = Tsr([[[0, 0, 1]]]).repeat(N, W, 1)  # shape: (N, W, Dtc)

        # kn window data to tensor
        kn = Tsr(self.date_series.kn_win)

        # create target data as `tg` (target)
        tg = self.dct_curve[self.model_type](ti).repeat(1, 1, self.Dout)

        ti, tc, kn, tg = self.to_device(ti, tc, kn, tg)
        trainset = DateTensors(
            ti=ti, tc=tc, kn=kn, tg=tg, device=self.device
        )  # ti/tc/kn.shape: (N, W, Dout), tg.shape = (N, 1, Dout)
        self.trainset = trainset
        return trainset

    def to_device(
        self, ti: Tsr, tc: Tsr, kn: Tsr, tg: Tsr
    ) -> Tuple[Tsr, Tsr, Tsr, Tsr]:
        device = self.device
        return ti.to(device), tc.to(device), kn.to(device), tg.to(device)

    def create_testset(self, by="2019-12-31") -> DateTensors:
        nxt = self.date_series.next_date
        test_series = DatasetDateSeries(
            start=nxt,
            end=by,
            wsz=self.date_series.wsz,
            wshift=self.date_series.wshift,
            to_onehot=True,
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
        testset = DateTensors(
            ti=ti, tc=tc, kn=kn, tg=tg, device=self.device
        )  # ti/tc/kn.shape: (N, W, Dout), tg.shape = (N, 1, Dout)
        self.testset = testset
        return testset


if __name__ == "__main__":
    # create toy dataset
    W, Dout = 14, 1
    toydataset = DatesetToy(Dout, W, "cycle", device=torch.device("cuda:0"))

    # test DateTensors
    ti = tv = tc = kn = tg = torch.arange(0, 60).view(4, 3, -1)
    batch = BatchMaker(bsz=2)
    for idx, bch in enumerate(zip(batch(ti), batch(tv), batch(tc), batch(kn))):
        print(f"{idx:02d}", type(bch), "n_zip:", len(bch), "bsz:", len(bch[0]))

    tsr = DateTensors(ti, tv, tc, kn, tg)
    for idx, bch in enumerate(tsr(bsz=2)):
        print(f"{idx:02d}", type(bch), "bsz:", len(bch))

    for idx, bch in enumerate(tsr(bsz=4)):
        print(f"{idx:02d}", type(bch), "bsz:", len(bch))
