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

import mlflow
import pickle


# Tsr = torch.DoubleTensor
Tsr = torch.Tensor


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

    def forward(self, ti: Tsr, tc: Tsr, kn: Tsr, un: Tsr, tg: Tsr):
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


class PositionalEncoding(nn.Module):
    """
    c.f. https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.0, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-numpy.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)        # (W, D) -> (1, W, D) -> (W, 1, D)
        self.register_buffer('pe', pe)

    def forward(self, x: Tsr) -> Tsr:
        """
        Shapes:
            x: (W, B, D)
            B: batch size
            W: window size
            D: dim
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)      # (W, B, D)


class Cyclic(M):
    def __init__(
        self,
        dim_ins: tuple,
        dim_out: int,
        ws: int,
        dim_emb=4*2,    # dim for embedding
        n_heads=4,
        k=5,            # the numbers of sin/cos curves
        n_layers=1,     # layers of multi-heads
        n_quantiles=7,
    ):
        super().__init__()  # after this call, to be enabled to access `self.args`
        n_dim = sum(dim_ins[1:])
        self.emb = nn.Linear(n_dim, dim_emb)    #.double()
        # TODO: implement using nn.Embedding
        ## nn.Embedding
        S = dim_ins[1]      # S: seq length
        max_len = max(16, S)
        self.pos = PositionalEncoding(d_model=dim_emb, max_len=max_len)

        # Transformers
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tr = nn.Transformer(**prm)    # .double()

        # linears
        self.ak = nn.Linear(ws * dim_emb, k * n_quantiles)  # .double()
        self.bk = nn.Linear(ws * dim_emb, k * n_quantiles)  # .double()
        self.wk = nn.Linear(ws * dim_emb, k)                # .double()
        self.ok = nn.Linear(ws * dim_emb, k)                # .double()

        # initialize weights
        weight_interval = 0.1
        nn.init.uniform_(self.emb.weight, -weight_interval, weight_interval)
        nn.init.kaiming_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.ok.weight)

        for fc in [self.emb, self.ak, self.bk, self.wk, self.ok]:
            nn.init.zeros_(fc.bias)

    def forward(
        self, ti: Tsr, tc: Tsr, kn: Tsr
    ):
        # embedding
        x = torch.cat([tc, kn], dim=-1)
        emb = self.emb(x) * numpy.sqrt(x.shape[-1])
        emb = emb.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb = self.pos(emb)

        # transform
        def reshape(tsr: Tsr) -> Tsr:
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.ws * self.args.dim_emb)

        z = reshape(self.tr.encoder(emb))

        a = self.ak(z)
        b = self.bk(z)

        pi = numpy.pi
        # w = torch.sigmoid(self.wk(z)) / (2 * pi)
        # o = torch.sigmoid(self.ok(z)) / pi
        # o = pi/2 + torch.sin(self.ok(z)) / (pi/2)      # map to (0, pi)
        w = self.wk(z)
        o = self.ok(z)

        # adjusting the shape
        k, q = self.args.k, self.args.n_quantiles
        kq = k * q
        t = ti[:, -1, :].repeat(1, kq).view(-1, k, q)
        w = w.view(-1, self.args.k, 1).repeat(1, 1, 7).view(-1, k, q)
        o = o.view(-1, self.args.k, 1).repeat(1, 1, 7).view(-1, k, q)
        a = a.view(-1, k, q)
        b = b.view(-1, k, q)

        # calculate theta (rad)
        th = 2 * pi * w * t + o
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
        n_layers=1,
        n_quantiles=7,
    ):
        super().__init__()  # after this call, to be enabled to access `self.args`
        self.emb = nn.Linear(sum(dim_ins[0:]), dim_emb)     # .double()
        # TODO: implement positional encoding
        S = dim_ins[1]      # S: seq length
        max_len = max(16, S)
        self.pos = PositionalEncoding(d_model=dim_emb, max_len=max_len)

        # Transformers
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tra = nn.Transformer(**prm)    # .double()
        self.trb = nn.Transformer(**prm)    # .double()
        self.tro = nn.Transformer(**prm)    # .double()

        # linears
        self.ak = nn.Linear(ws * dim_emb, k)            # .double()
        self.bk = nn.Linear(ws * dim_emb, n_quantiles)  # .double()
        self.ok = nn.Linear(ws * dim_emb, k)            # .double()
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.ok.weight)

    def forward(
        self, ti: Tsr, tc: Tsr, kn: Tsr
    ):
        # embedding
        x = torch.cat([ti, tc, kn], dim=-1)
        emb = self.emb(x) * numpy.sqrt(x.shape[-1])
        emb = emb.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb = self.pos(emb)

        # transform
        def reshape(tsr: Tsr):
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

    def get_quantile(self, x: Tsr, alpha: float):
        idx = (numpy.array(self.params.quantiles) == alpha).argmax()
        return x[:, idx, :][..., 0]  # just to get a first dim

    def do_pretrain(self, dataset, epochs: int = 500):
        # use mask
        # swap seqence
        pass

    def do_train(self, dataset, epochs: int = 500):
        ti, tc, kn, tg = dataset.trainset.ti, dataset.trainset.tc, dataset.trainset.kn, dataset.trainset.tg
        batch = BatchMaker(bsz=self.params.batch_size)

        # train loop
        n_steps = -1
        losses = []
        # # epoch loop
        for idx in range(epochs):
            shuffle = numpy.random.permutation(range(len(ti)))
            _ti, _tc, _kn, _tg = ti[shuffle], tc[shuffle], kn[shuffle], tg[shuffle]

            # # batch loop
            for bdx, (bti, btc, bkn, btg) in enumerate(
                zip(batch(_ti), batch(_tc), batch(_kn), batch(_tg))
            ):
                n_steps += 1

                def closure():
                    self.optimizer.zero_grad()
                    y_pred = self.model(bti, btc, bkn)
                    loss = self.criterion(y_pred, btg[:, -1, :], self.params.quantiles)
                    losses.append(loss.item())
                    assert len(losses) == n_steps + 1

                    self.loss_train = loss.item()  # keep latest loss
                    k = self.params.log_interval
                    l = loss.item() if n_steps < k else numpy.mean(losses[-k :])
                    self.writer.add_scalar(
                        f"loss/step/train", l, n_steps
                    )

                    # logging progress
                    if idx % self.params.log_interval == 0 and idx > 0 and bdx == 0:
                        mean_loss = numpy.mean(losses[-k :])
                        print(f"loss[{idx:03d}][{bdx:03d}][{n_steps:05d}]", mean_loss)

                        def make_predictions(y_pred, tg) -> Tuple[Tsr, Tsr, Tsr, Tsr]:
                            pred = y_pred.view(
                                -1, len(self.params.quantiles), self.model.args.dim_out
                            )
                            p = self.get_quantile(pred, alpha=0.5)
                            p10 = self.get_quantile(pred, alpha=0.1)
                            p90 = self.get_quantile(pred, alpha=0.9)
                            t = tg[:, -1, :][..., 0]
                            return p, p10, p90, t


                        def write_log2tb(idx, preds, loss, pred_type="train") -> None:
                            for n, (y0, yL, yH, t0) in enumerate(zip(*preds)):
                                dct_pred = dict(p=y0, p10=yL, p90=yH, t=t0)
                                self.writer.add_scalars(
                                    f"prediction/epoch_{idx:03d}/{pred_type}",
                                    dct_pred,
                                    n,
                                )
                            self.writer.add_scalar(
                                f"loss/interval/{pred_type}", loss.item(), idx
                            )

                        def do_predict(idx, ti: Tsr, tc: Tsr, kn: Tsr, tg: Tsr, pred_mode="train") -> Tsr:
                            bti, btc, bkn, btg = next(batch(ti)), next(batch(tc)), next(batch(kn)), next(batch(tg))
                            with torch.no_grad():
                                pred = self.model(bti, btc, bkn)
                                loss = self.criterion(
                                    pred, btg[:, -1, :], self.params.quantiles
                                )
                            preds = make_predictions(pred, btg)
                            write_log2tb(idx, preds, loss, pred_mode)
                            return loss

                        # prediction with trainset
                        loss_train = do_predict(idx, ti, tc, kn, tg, pred_mode="train")
                        self.loss_train = loss_train.item()

                        # prediction with testset
                        testset = dataset.create_testset()
                        loss_valid = do_predict(idx, testset.ti, testset.tc, testset.kn, testset.tg, pred_mode="valid")
                        self.loss_valid = loss_valid.item()

                    loss.backward()
                    return loss

                self.optimizer.step(closure)

            if idx % self.params.save_interval == 0 and idx > 0:
                mlflow.pytorch.log_model(self.model, f"models.{idx:05d}", pickle_module=pickle)

def quantile_loss(
    pred_y: Tsr,
    tg: Tsr,
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    with_mse=False,
) -> Tsr:

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
    pred_y: Tsr,
    tg: Tsr,
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
) -> Tsr:
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
        default=30 * 1000,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from ..dataset.dateset import DatesetToy

    args = get_args()

    # setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # create toy dataset
    B, W, Dout = 64, 14, 1
    toydataset = DatesetToy(Dout, W, args.model, device=device)
    trainset = toydataset.create_trainset()

    # setup criterion
    criterionner = dict(cyclic=quantile_loss, trend=quantile_loss_with_mse)
    criterion = criterionner[args.model]

    if args.resume:
        print("loading the latest model ...")
        model = mlflow.pytorch.load_model(f"models:/latest_model/1")
        print("loading done.")
    else:
        # setup model
        dims = (trainset.ti.shape[-1], trainset.tc.shape[-1], trainset.kn.shape[-1])
        modeller = dict(cyclic=Cyclic, trend=Trend)
        model = modeller[args.model](
            dim_ins=dims,
            dim_out=trainset.tg.shape[-1],
            ws=trainset.ti.shape[1],
            dim_emb=8,
            n_heads=4,
            k=3,
            n_layers=1,
            n_quantiles=len(criterion.__defaults__[0]), # like len(quantiles)
        )
    model.to(device)
    print("model.args:", model.args)

    # setup optimizer and criterion
    # optimizer = optim.LBFGS(model.parameters(), lr=0.8)     # Newton
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # setup tensorboard writer
    experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = f"result/{experiment_id}"
    writer = SummaryWriter(log_dir=logdir)

    params = Items(is_readonly=True).setup(
        dict(
            batch_size=B,
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        )
    )
    trainer = Trainer(model, optimizer, criterion, writer, params)
    writer.add_graph(model, (trainset.ti, trainset.tc, trainset.kn))

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

    mlflow.pytorch.log_model(model, "models", registered_model_name="latest_model", pickle_module=pickle)