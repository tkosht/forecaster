import torch
from typing import Tuple
from .dateseries import DatasetDateData


class DatasetToy(object):
    def __init__(self, Dout: int, model_type: str, device: torch.device):
        self.Dout = Dout
        self.model_type = model_type
        self.device = device
        self.dateset = None

        self.ti = torch.Tensor([])
        self.tc = torch.Tensor([])
        self.kn = torch.Tensor([])
        self.tg = torch.Tensor([])

    def create(
        self,
        s="2016-01-01",
        e="2018-12-31",
        freq="D",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ti.shape = (N, W, 1)
        """
        self.dateset = DatasetDateData(start=s, end=e, wsz=7)
        self.ti = torch.DoubleTensor(self.dateset.seqti)
        Dtc = 3
        N, W = len(self.ti), self.dateset.wsz
        self.tc = torch.randint(0, 2, (1, 1, Dtc)).repeat(N, W, 1)  # shape: (B, W, Dtc)
        self.kn = torch.DoubleTensor(self.dateset.seqdata)
        # create target data as `self.tg`
        if self.model_type == "cyclic":
            # self.tg = torch.sin(ti).repeat(1, 1, Dout)  # target
            self.tg = (
                torch.sin(2 * self.ti)
                + 3 * torch.sin(self.ti)
                + 0.5 * torch.cos(3 * self.ti)
            ).repeat(1, 1, self.Dout)
        elif self.model_type == "trend":
            self.tg = (0.0001 * self.ti ** 2 + 0.005 * self.ti + 2).repeat(
                1, 1, self.Dout
            )
        else:
            raise NotImplementedError(f"{self.__class__}.create()")
        self.ti, self.tc, self.kn, self.tg = self.to_device(self.ti, self.tc, self.kn, self.tg)
        return self.ti, self.tc, self.kn, self.tg

    def to_device(self, ti: torch.DoubleTensor, tc: torch.DoubleTensor, kn: torch.DoubleTensor, tg: torch.DoubleTensor) -> Tuple[torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor]:
        device = self.device
        return ti.to(device), tc.to(device), kn.to(device), tg.to(device)

    def create_testset(
        self, by="2019-12-31"
    ) -> Tuple[torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor]:
        nxt = self.dateset.next_date
        dateset_test = DatasetDateData(
            start=nxt, end=by, wsz=self.dateset.wsz, wshift=self.dateset.wshift
        )
        test_ti = torch.DoubleTensor(dateset_test.seqti)
        N = len(test_ti)
        W = dateset_test.wsz
        test_tc = (
            self.tc[0, 0, :].reshape(1, 1, -1).repeat(N, W, 1)
        )  # shape: (N, W, Dtc)
        test_kn = torch.DoubleTensor(dateset_test.seqdata)
        if self.model_type == "cyclic":
            test_tg = (
                torch.sin(2 * test_ti)
                + 3 * torch.sin(test_ti)
                + 0.5 * torch.cos(3 * test_ti)
            ).repeat(1, 1, self.Dout)
        elif self.model_type == "trend":
            test_tg = (0.0001 * test_ti ** 2 + 0.005 * test_ti + 2).repeat(
                1, 1, self.Dout
            )
        else:
            raise NotImplementedError(f"{self.__class__}.create_testset()")
        test_ti, test_tc, test_kn, test_tg = self.to_device(test_ti, test_tc, test_kn, test_tg)
        return test_ti, test_tc, test_kn, test_tg
