import inspect
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from ..util.items import Items


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.set_params()

    def set_params(self):
        _frame = inspect.currentframe()
        _locals = _frame.f_back.f_back.f_locals
        assert _frame.f_back.f_back.f_locals["__class__"] == type(self)
        args = {k: v for k, v in _locals.items() if k not in ["self"] and k[:1] != "_"}
        self.args = Items().setup(args)


class Model(M):
    def __init__(self, dim_ins=16, n_heads=4, ws=8, n_quantiles=7):
        super().__init__()  # after this call, to be enabled to access `self.args`
        self.cyclic = Cyclic(
            dim_ins=dim_ins, n_heads=4, n_quantiles=self.args.n_quantiles
        )
        self.trend = None
        self.recent = None
        self.ws = ws  # window size

    def forward(
        self,
        ti: torch.Tensor,
        tc: torch.Tensor,
        kn: torch.Tensor,
        un: torch.Tensor,
        tg: torch.Tensor,
    ):
        r"""
        Args:
            ti: time index
            tc: time constant
            kn: known time variant / exog
            un: unknown time variant / multi variable
        Shapes:
            ti: (B, W, D1)
            tc: (B, W, D2)
            kn: (B, W, D3)
            un: (B, W, D4)
            B: batch size
            W: window size
            D?: dims
        """
        cyclic = self.cyclic(ti, tc, kn, tg)
        trend = self.trend(ti, tc, kn, tg)
        ws = self.ws
        recent = self.recent(ti, kn, tc[:, -ws:, :], un[:, -ws:, :], tg)
        y = cyclic + trend + recent
        return y


class Cyclic(M):
    def __init__(
        self,
        dim_ins: tuple,
        dim_out: int,
        ws: int,
        dim_emb=5,
        n_heads=4,
        k=3,
        n_quantiles=7,
        n_layers=2,
    ):
        super().__init__()  # after this call, to be enabled to access `self.args`
        self.emb = nn.Linear(sum(dim_ins[1:]), dim_emb)

        # Transformers
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tra = nn.Transformer(**prm)
        self.trb = nn.Transformer(**prm)
        self.trw = nn.Transformer(**prm)
        self.tro = nn.Transformer(**prm)

        # linears
        self.ak = nn.Linear(dim_emb * ws, k * n_quantiles)
        self.bk = nn.Linear(dim_emb * ws, k * n_quantiles)
        self.wk = nn.Linear(dim_emb * ws, k)
        self.ok = nn.Linear(dim_emb * ws, k)

        # initialize weights
        nn.init.kaiming_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.ok.weight)

    def forward(self, ti: torch.Tensor, tc: torch.Tensor, kn: torch.Tensor):
        # embedding
        x = torch.cat([tc, kn], dim=-1)
        emb = self.emb(x)
        emb = emb.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)

        # transform
        def reshape(tsr: torch.Tensor):
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.dim_emb * self.args.ws)

        ha = reshape(self.tra.encoder(emb))
        hb = reshape(self.trb.encoder(emb))
        hw = reshape(self.trw.encoder(emb))
        ho = reshape(self.tro.encoder(emb))
        a = self.ak(ha)
        b = self.bk(hb)
        w = torch.sigmoid(self.wk(hw)) / (2 * numpy.pi)
        o = torch.sigmoid(self.ok(ho)) / numpy.pi

        # adjusting the shape
        k, q = self.args.k, self.args.n_quantiles
        kq = k * q
        t = ti[:, -1, :].repeat(1, kq).view(-1, k, q)
        w = w.view(-1, self.args.k, 1).repeat(1, 1, 7).view(-1, k, q)
        o = o.view(-1, self.args.k, 1).repeat(1, 1, 7).view(-1, k, q)
        a = a.view(-1, k, q)
        b = b.view(-1, k, q)

        # calculate theta (rad)
        th = 2 * numpy.pi * w * t + o
        y = a * torch.cos(th) + b * torch.sin(th)
        y = y.sum(dim=1)

        return y


class Trend(M):
    def __init__(
        self,
        dim_ins: tuple,
        dim_out: int,
        ws: int,
        dim_emb=3,
        n_heads=3,
        k=3,
        n_quantiles=7,
        n_layers=1,
    ):
        super().__init__()  # after this call, to be enabled to access `self.args`
        self.emb = nn.Linear(sum(dim_ins[1:]), dim_emb)

        # Transformers
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tra = nn.Transformer(**prm)
        self.trb = nn.Transformer(**prm)
        self.tro = nn.Transformer(**prm)

        # linears
        self.ak = nn.Linear(dim_emb * ws, k)
        self.bk = nn.Linear(dim_emb * ws, n_quantiles)
        self.ok = nn.Linear(dim_emb * ws, k)
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.ok.weight)

    def forward(self, ti: torch.Tensor, tc: torch.Tensor, kn: torch.Tensor):
        # embedding
        x = torch.cat([tc, kn], dim=-1)
        emb = self.emb(x)
        emb = emb.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)

        # transform
        def reshape(tsr: torch.Tensor):
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.dim_emb * self.args.ws)

        ha = reshape(self.tra.encoder(emb))
        hb = reshape(self.trb.encoder(emb))
        ho = reshape(self.tro.encoder(emb))
        a = 2 * torch.tanh(self.ak(ha))
        b = self.bk(hb)
        o = self.ok(ho)

        # adjusting the shape
        k, q = self.args.k, self.args.n_quantiles
        kq = k * q
        t = ti[:, -1, :].repeat(1, kq).view(-1, k, q)  # shape = (B, k, q)
        a = a.view(-1, k, 1).repeat(1, 1, q)  # (B, k, q)
        b = b.view(-1, 1, q).repeat(1, k, 1)  # (B, k, q)
        o = o.view(-1, k, 1).repeat(1, 1, q)  # (B, k, q)

        # calculate trend line with t
        y = a * self.relu(t - o) + b
        y = y.sum(dim=1)  # ¥sum_k y_{b, k, q}

        return y


class Trainer(object):
    def __init__(self, model, optimizer, criterion, writer, params):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.params = params
        self.loss_train = None
        self.loss_valid = None

    def get_quantile(self, x: torch.Tensor, alpha: float):
        idx = (numpy.array(self.params.quantiles) == alpha).argmax()
        return x[:, idx, :][..., 0]  # just to get a first dim

    def do_train(self, dataset: Items, epochs: int = 500):
        ti, tc, kn, tg = dataset.ti, dataset.tc, dataset.kn, dataset.tg
        # train loop
        batch_losses = []
        for idx in range(epochs):
            shuffle = numpy.random.permutation(range(len(ti)))
            t = ti[shuffle]

            def closure():
                self.optimizer.zero_grad()
                y_pred = self.model(t, tc, kn)
                loss = self.criterion(y_pred, tg[:, -1, :], self.params.quantiles)
                batch_losses.append(loss.item())

                # logging progress
                if idx % self.params.log_intervals == 0 and idx > 0:
                    mean_loss = numpy.array(batch_losses[idx - 20 : idx]).mean()
                    print(f"loss[{idx:03d}]", mean_loss)
                    self.writer.add_scalar("loss/train", loss.item(), idx)
                    self.loss_train = loss.item()

                    def make_predictions(
                        y_pred, tg
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                        pred = y_pred.view(
                            -1, len(self.params.quantiles), self.model.args.dim_out
                        )
                        p = self.get_quantile(pred, alpha=0.5)
                        p10 = self.get_quantile(pred, alpha=0.1)
                        p90 = self.get_quantile(pred, alpha=0.9)
                        t = tg[:, -1, :][..., 0]
                        return p, p10, p90, t

                    def write_log2tb(preds, loss, pred_type="train") -> None:
                        for n, (y0, yL, yH, t0) in enumerate(zip(*preds)):
                            dct_pred = dict(p=y0, p10=yL, p90=yH, t=t0)
                            self.writer.add_scalars(
                                f"prediction/epoch_{idx}/{pred_type}", dct_pred, n
                            )
                        self.writer.add_scalar(f"loss/{pred_type}", loss.item(), idx)

                    # prediction with trainset
                    preds = make_predictions(y_pred, tg)
                    write_log2tb(preds, loss, "train")

                    # prediction with testset
                    test_ti, test_tc, test_kn, test_tg = toydataset.create_testset()
                    with torch.no_grad():
                        _y_pred = self.model(test_ti, test_tc, test_kn)
                        loss_valid = self.criterion(
                            _y_pred, test_tg[:, -1, :], self.params.quantiles
                        )
                    preds = make_predictions(_y_pred, test_tg)
                    write_log2tb(preds, loss_valid, "valid")
                    self.loss_valid = loss_valid.item()

                loss.backward()
                return loss

            self.optimizer.step(closure)

        self.writer.close()


class DatasetToy(object):
    def __init__(self, B: int, W: int, Dout: int, model: str):
        self.B = B
        self.W = W
        self.Dout = Dout
        self.model = model
        self.ti = torch.Tensor([])
        self.tc = torch.Tensor([])
        self.kn = torch.Tensor([])
        self.tg = torch.Tensor([])

    def create(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ti = [  [[1], [2], [3], [4]],
                [[2], [3], [4], [5]],
                [[3], [4], [5], [6]],
                [[4], [5], [6], [7]],
                [[5], [6], [7], [8]],
                :
                ]
        ti.shape = (B, W, 1)
        """
        B, W, Dout = self.B, self.W, self.Dout
        t0 = torch.arange(0, W).view(1, W, Dout)
        ti = t0
        for idx in range(B - 1):
            t1 = t0 + idx + 1
            ti = torch.cat([ti, t1], axis=0)
        self.ti = ti.float()
        Dtc, Dkn = 3, 4
        self.tc = torch.randint(0, 2, (1, 1, Dtc)).repeat(B, W, 1)  # shape: (B, W, Dtc)
        self.kn = torch.randn((1, 1, Dkn)).repeat(B, W, 1)  # shape: (B, W, Dkn)
        if self.model == "cyclic":
            # self.tg = torch.sin(ti).repeat(1, 1, Dout)  # target
            self.tg = (
                torch.sin(2 * self.ti)
                + 3 * torch.sin(self.ti)
                + 0.5 * torch.cos(3 * self.ti)
            ).repeat(1, 1, Dout)
        elif self.model == "trend":
            self.tg = (0.0001 * self.ti ** 2 + 0.005 * self.ti + 2).repeat(1, 1, Dout)
        else:
            raise NotImplementedError(f"{self.__class__}.create()")
        return self.ti, self.tc, self.kn, self.tg

    def create_testset(
        self, offset=10, sz=32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ti, tc, kn, tg = self.ti, self.tc, self.kn, self.tg
        test_ti = (ti + len(ti))[offset : offset + sz]
        test_tc = tc[offset : offset + sz]
        test_kn = kn[offset : offset + sz]
        if self.model == "cyclic":
            test_tg = tg[offset : offset + sz]
        elif self.model == "trend":
            test_tg = (0.0001 * test_ti ** 2 + 0.005 * test_ti + 2).repeat(
                1, 1, self.Dout
            )
        else:
            raise NotImplementedError(f"{self.__class__}.create_testset()")
        return test_ti, test_tc, test_kn, test_tg


def quantile_loss(
    pred_y: torch.Tensor,
    tg: torch.Tensor,
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    with_mse=False,
) -> torch.Tensor:

    losses = []
    for idx, qtl in enumerate(quantiles):
        err = tg - pred_y[..., idx].unsqueeze(-1)  # (B, 1)
        losses.append(torch.max((qtl - 1) * err, qtl * err).unsqueeze(-1))  # (B, 1, 1)
    losses = torch.cat(losses, dim=2)
    loss = losses.sum(dim=(1, 2)).mean()
    if with_mse:
        loss += nn.MSELoss()(pred_y, tg.repeat(1, len(quantiles)))
    return loss


def quantile_loss_with_mse(
    pred_y: torch.Tensor,
    tg: torch.Tensor,
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
) -> torch.Tensor:
    return quantile_loss(pred_y, tg, quantiles, with_mse=True)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Model Toy Test")
    parser.add_argument(
        "--model",
        type=str,
        choices=["cyclic", "trend", "recent", "full"],
        default="cyclic",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=10 * 1000,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import datetime

    args = get_args()

    # create toy dataset
    B, W, Dout = 64, 4, 1
    toydataset = DatasetToy(B, W, Dout, args.model)
    ti, tc, kn, tg = toydataset.create()
    dataset = Items(is_readonly=True).setup(dict(ti=ti, tc=tc, kn=kn, tg=tg))

    # setup model
    dims = (ti.shape[-1], tc.shape[-1], kn.shape[-1])
    modeller = dict(cyclic=Cyclic, trend=Trend)
    model = modeller[args.model](
        dim_ins=dims,
        dim_out=tg.shape[-1],
        ws=ti.shape[1],
        dim_emb=4 * 2,
        n_heads=2,
        k=3,
        n_quantiles=7,
    )

    # setup optimizer and criterion
    # optimizer = optim.LBFGS(model.parameters(), lr=0.8)     # Newton
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterionner = dict(cyclic=quantile_loss, trend=quantile_loss_with_mse)
    criterion = criterionner[args.model]

    # setup tensorboard writer
    experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = f"result/{experiment_id}"
    writer = SummaryWriter(log_dir=logdir)

    params = Items(is_readonly=True).setup(
        dict(
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            log_intervals=100,
        )
    )
    trainer = Trainer(model, optimizer, criterion, writer, params)
    writer.add_graph(model, (ti, tc, kn))

    # train
    trainer.do_train(dataset, epochs=args.max_epoch)

    # experiment log
    hp = dict(
        experiment_id=experiment_id,
        model=args.model,
        max_epoch=args.max_epoch,
        optimizer=str(optimizer),
        criterion=str(criterion),
    )
    writer.add_hparams(
        hp,
        {
            "hparam/loss/train": trainer.loss_train,
            "hparam/loss/valid": trainer.loss_valid,
        },
    )
