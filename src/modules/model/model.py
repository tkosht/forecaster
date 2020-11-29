import inspect
import datetime
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from ..util.items import Items
from ..dataset.batcher import BatchMaker


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
        self.emb = nn.Linear(sum(dim_ins[1:]), dim_emb).double()

        # Transformers
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tra = nn.Transformer(**prm).double()
        self.trb = nn.Transformer(**prm).double()
        self.trw = nn.Transformer(**prm).double()
        self.tro = nn.Transformer(**prm).double()

        # linears
        self.ak = nn.Linear(dim_emb * ws, k * n_quantiles).double()
        self.bk = nn.Linear(dim_emb * ws, k * n_quantiles).double()
        self.wk = nn.Linear(dim_emb * ws, k).double()
        self.ok = nn.Linear(dim_emb * ws, k).double()

        # initialize weights
        nn.init.kaiming_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.ok.weight)

    def forward(
        self, ti: torch.DoubleTensor, tc: torch.DoubleTensor, kn: torch.DoubleTensor
    ):
        # embedding
        x = torch.cat([tc, kn], dim=-1)
        emb = self.emb(x)
        emb = emb.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)

        # transform
        def reshape(tsr: torch.DoubleTensor):
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
        self.emb = nn.Linear(sum(dim_ins[0:]), dim_emb).double()

        # Transformers
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tra = nn.Transformer(**prm).double()
        self.trb = nn.Transformer(**prm).double()
        self.tro = nn.Transformer(**prm).double()

        # linears
        self.ak = nn.Linear(dim_emb * ws, k).double()
        self.bk = nn.Linear(dim_emb * ws, n_quantiles).double()
        self.ok = nn.Linear(dim_emb * ws, k).double()
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.ok.weight)

    def forward(
        self, ti: torch.DoubleTensor, tc: torch.DoubleTensor, kn: torch.DoubleTensor
    ):
        # embedding
        x = torch.cat([ti, tc, kn], dim=-1)
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
    def __init__(self, model, optimizer, criterion, writer, params: Items):
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

    def do_train(self, dataset, epochs: int = 500):
        ti, tc, kn, tg = dataset.ti, dataset.tc, dataset.kn, dataset.tg
        batch = BatchMaker(bsz=self.params.batch_size)
        # train loop
        n_steps = -1
        losses = []
        for idx in range(epochs):
            shuffle = numpy.random.permutation(range(len(ti)))
            _ti, _tc, _kn, _tg = ti[shuffle], tc[shuffle], kn[shuffle], tg[shuffle]

            for bdx, (bti, btc, bkn, btg) in enumerate(
                zip(batch(_ti), batch(_tc), batch(_kn), batch(_tg))
            ):
                n_steps += 1

                def closure():
                    self.optimizer.zero_grad()
                    y_pred = self.model(bti, btc, bkn)
                    loss = self.criterion(y_pred, btg[:, -1, :], self.params.quantiles)
                    losses.append(loss.item())

                    self.loss_train = loss.item()  # keep latest loss

                    # logging progress
                    if idx % self.params.log_interval == 0 and idx > 0 and bdx == 0:
                        mean_loss = numpy.array(losses[n_steps - 20 : n_steps]).mean()
                        print(f"loss[{idx:03d}][{bdx:03d}][{n_steps:05d}]", mean_loss)

                        def make_predictions(
                            y_pred, tg
                        ) -> Tuple[
                            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
                        ]:
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
                                    f"prediction/epoch_{idx:03d}/{pred_type}",
                                    dct_pred,
                                    n,
                                )
                            self.writer.add_scalar(
                                f"loss/{pred_type}", loss.item(), idx
                            )

                        # prediction with trainset
                        preds = make_predictions(y_pred, tg)
                        write_log2tb(preds, loss, "train")

                        # prediction with testset
                        test_ti, test_tc, test_kn, test_tg = dataset.create_testset()
                        test_bti = next(batch(test_ti))
                        test_btc = next(batch(test_tc))
                        test_bkn = next(batch(test_kn))
                        test_btg = next(batch(test_tg))
                        with torch.no_grad():
                            test_pred = self.model(test_bti, test_btc, test_bkn)
                            loss_valid = self.criterion(
                                test_pred, test_btg[:, -1, :], self.params.quantiles
                            )
                        preds = make_predictions(test_pred, test_btg)
                        write_log2tb(preds, loss_valid, "valid")
                        self.loss_valid = loss_valid.item()

                    loss.backward()
                    return loss

                self.optimizer.step(closure)


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
        default=100,  # 10 * 1000
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,  # 100
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from ..dataset.dateset import DatasetToy

    args = get_args()

    # setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # create toy dataset
    B, W, Dout = 64, 4, 1
    toydataset = DatasetToy(Dout, args.model, device=device)
    ti, tc, kn, tg = toydataset.create()

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
    ).to(device)

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
            batch_size=64,
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            log_interval=args.log_interval,
        )
    )
    trainer = Trainer(model, optimizer, criterion, writer, params)
    writer.add_graph(model, (ti, tc, kn))

    # train
    trainer.do_train(toydataset, epochs=args.max_epoch)

    # experiment log
    hparams = dict(
        experiment_id=experiment_id,
        model=args.model,
        max_epoch=args.max_epoch,
        optimizer=str(optimizer),
        criterion=str(criterion),
    )
    writer.add_hparams(
        hparams,
        {
            "hparam/loss/train": trainer.loss_train,
            "hparam/loss/valid": trainer.loss_valid,
        },
    )
    writer.close()
