import inspect
import numpy
import pandas
import torch
import torch.nn as nn
from ..util.items import Items
from ..dataset.dateset import Tsr
from ..loss import loss_quantile, loss_mse

# Tsr = torch.DoubleTensor
# Tsr = torch.Tensor


class ModelBase(nn.Module):
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    name = "base"

    def __init__(
        self,
        dim_ins: tuple,  # dims of ti, tc, kn
        dim_out: int,  # just reserved
        ws: int,  # windows size of time series/sequence
        dim_emb: int,  # dim for embedding
        n_heads: int,  # the number of attention heads in transformer
        n_layers: int,  # layers of multi-heads
        k: int,  # the numbers of curves
    ):
        super().__init__()
        self.set_params()

        # override args from child/sub class
        dim_ins = self.args.dim_ins
        dim_emb = self.args.dim_emb
        ws = self.args.ws
        n_heads = self.args.n_heads
        n_layers = self.args.n_layers

        self.n_quantiles = len(self.quantiles)  # the number of quantiles

        # embedder
        n_dim = sum(dim_ins[0:])
        self.emb_encode = nn.Linear(n_dim, dim_emb)  # .double()
        self.emb_decode = nn.Linear(n_dim, dim_emb)  # .double()
        max_len = max(16, ws)
        self.pos = PositionalEncoding(d_model=dim_emb, max_len=max_len)

        # Transformer
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tr = nn.Transformer(**prm)  # .double()

        # linears
        self.dc = nn.Linear(ws * dim_emb, ws * n_dim * self.n_quantiles)  # .double()

        # constraint
        self.loss_constraint_pretrain = 0

        # to double
        self.emb_encode = self.emb_encode.double()
        self.emb_decode = self.emb_decode.double()
        self.tr = self.tr.double()
        self.dc = self.dc.double()

        # initialize weights/biases
        weight_interval = 0.01
        nn.init.uniform_(self.emb_encode.weight, -weight_interval, weight_interval)
        nn.init.uniform_(self.emb_decode.weight, -weight_interval, weight_interval)
        nn.init.xavier_normal_(self.dc.weight)
        for fc in [self.emb_encode, self.emb_decode, self.dc]:
            nn.init.zeros_(fc.bias)

    def set_params(self):
        _frame = inspect.currentframe()
        _locals = _frame.f_back.f_back.f_locals
        assert _frame.f_back.f_back.f_locals["__class__"] == type(self)
        args = {k: v for k, v in _locals.items() if k not in ["self"] and k[:1] != "_"}
        self.args = Items().setup(args)

    def make_x(self, ti: Tsr, tc: Tsr, kn: Tsr):
        ti_base = pandas.date_range("2010-1-1", "2010-1-2")[0]
        ti_base = ti_base.to_numpy().astype(numpy.int64)
        n_digits = numpy.int64(numpy.floor(numpy.log10(ti_base)))
        interval = (
            numpy.sqrt(2) * (10 ** n_digits) / 10
        )  # make interval to be irrational number
        _ti = ti / interval  # _ti scales to (0, 1) by 2200/1/1
        x = torch.cat([_ti, tc, kn], dim=-1)  # as input
        return x

    def _pretrain(self, ti: Tsr, tc: Tsr, kn: Tsr) -> Tsr:
        # setup
        mask_rate = 0.15

        # concat input
        x = self.make_x(ti, tc, kn)  # as input
        y = x.clone()  # as target

        # make mask
        B, W, D = x.shape
        zr = torch.zeros((1, W, D))
        mask_vector = -1 * torch.ones_like(zr)  # (1, W, D)
        probs = mask_rate * torch.ones(1, W, 1)  # (1, W, 1)
        msk = torch.bernoulli(probs).repeat(1, 1, D)  # (1, W, D)
        mask_vector *= msk  # (1, W, D)
        mask_vector = mask_vector.repeat(B, 1, 1)  # this vector means `[MASK]`
        msk_flipped = 1 - msk.type(torch.bool).type(torch.long)  # (1, W, D)
        assert (msk[0, :, 0] + msk_flipped[0, :, 0] == torch.ones(W)).all().item()

        # masking
        x *= msk_flipped.repeat(B, 1, 1).to(x.device)
        x += mask_vector.to(x.device)

        # embedding
        emb_encode = self.emb_encode(x) * numpy.sqrt(x.shape[-1])
        emb_encode = emb_encode.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb_encode = self.pos(emb_encode)

        # - for y
        emb_decode = self.emb_decode(y) * numpy.sqrt(y.shape[-1])
        # emb_decode = self.emb_encode(y) * numpy.sqrt(y.shape[-1])
        emb_decode = emb_decode.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb_decode = self.pos(emb_decode)

        # transform
        def reshape(tsr: Tsr) -> Tsr:
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.ws * self.args.dim_emb)

        encoded = reshape(self.tr(emb_encode, emb_decode))
        p = self.dc(encoded)
        p = p.reshape(*x.shape, self.n_quantiles)
        p = torch.sigmoid(p)

        return p

    def pretrain(self, ti: Tsr, tc: Tsr, kn: Tsr):
        raise NotImplementedError(type(self))

    def forward(self, ti: Tsr, tc: Tsr, kn: Tsr):
        raise NotImplementedError(type(self))

    def loss_pretrain(self, ti: Tsr, tc: Tsr, kn: Tsr, **params) -> Tsr:
        x = self.make_x(ti, tc, kn)
        y = x.unsqueeze(-1)
        p = self.pretrain(ti, tc, kn)  # (B, W, D, k)
        loss = self.calc_loss_pretrain(p, y, **params)
        return loss

    def calc_loss_pretrain(self, pred_y: Tsr, tg: Tsr, **params) -> Tsr:
        loss = loss_quantile(pred_y, tg)
        loss += self.calc_loss_pretrain_constraint(**params)
        return loss

    def calc_loss_pretrain_constraint(self, **params) -> Tsr:
        return 0.0

    def loss_train(self, ti: Tsr, tc: Tsr, kn: Tsr, tg: Tsr, **params) -> Tsr:
        p = self(ti, tc, kn)  # (B, W, D, k)
        loss = self.calc_loss(p, tg, **params)
        return loss

    def calc_loss(self, pred_y: Tsr, tg: Tsr, **params) -> Tsr:
        raise NotImplementedError(type(self))


class ModelTimesries(ModelBase):
    r"""
    TODO: to implement
    """

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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-numpy.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (W, D) -> (1, W, D) -> (W, 1, D)
        self.register_buffer("pe", pe)

    def forward(self, x: Tsr) -> Tsr:
        """
        Shapes:
            x: (W, B, D)
            B: batch size
            W: window size
            D: dim
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)  # (W, B, D)


class Cyclic(ModelBase):
    name = "cyclic"

    def __init__(
        self,
        dim_ins: tuple,
        dim_out: int,
        ws: int,
        dim_emb=4 * 2,  # dim for embedding
        n_heads=4,
        n_layers=1,  # layers of multi-heads
        k=5,  # the numbers of sin/cos curves
    ):
        super().__init__(
            dim_ins,
            dim_out,
            ws,
            dim_emb,
            n_heads,
            n_layers,
            k,
        )  # after this call, to be enabled to access `self.args`

        # linears
        self.ak = nn.Linear(ws * dim_emb, k * self.n_quantiles).double()
        self.bk = nn.Linear(ws * dim_emb, k * self.n_quantiles).double()
        self.wk = nn.Linear(ws * dim_emb, k).double()
        self.ok = nn.Linear(ws * dim_emb, k).double()

        # constraint
        self.loss_constraint = 0  # just reserved

        # initialize weights
        nn.init.kaiming_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.ok.weight)

        for fc in [self.ak, self.bk, self.wk, self.ok]:
            nn.init.zeros_(fc.bias)

    def make_x(self, ti: Tsr, tc: Tsr, kn: Tsr):
        pi = numpy.pi
        _ti = ti % (2 * pi) / (2 * pi)
        x = torch.cat([_ti, tc, kn], dim=-1)  # as input
        return x

    def pretrain(self, ti: Tsr, tc: Tsr, kn: Tsr, **params) -> Tsr:
        return self._pretrain(ti, tc, kn)

    def _forward(self, ti: Tsr, tc: Tsr, kn: Tsr):
        # embedding
        # - for X
        x = self.make_x(ti, tc, kn)  # as input
        emb_encode = self.emb_encode(x) * numpy.sqrt(x.shape[-1])
        emb_encode = emb_encode.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb_encode = self.pos(emb_encode)

        # transform
        def reshape(tsr: Tsr) -> Tsr:
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.ws * self.args.dim_emb)

        z = reshape(self.tr.encoder(emb_encode))

        a = self.ak(z)
        b = self.bk(z)

        w = self.wk(z)
        o = self.ok(z)

        pi = numpy.pi
        o = 2 * pi * torch.sigmoid(o)

        return x, a, b, w, o

    def forward(self, ti: Tsr, tc: Tsr, kn: Tsr):
        x, a, b, w, o = self._forward(ti, tc, kn)

        # adjusting the shape
        k, q = self.args.k, self.n_quantiles
        kq = k * q
        dim_ti = self.args.dim_ins[0]
        _ti = x[:, :, :dim_ti]
        t = _ti[:, -1, :].repeat(1, kq).view(-1, k, q)
        w = w.view(-1, k, 1).repeat(1, 1, q).view(-1, k, q)
        o = o.view(-1, k, 1).repeat(1, 1, q).view(-1, k, q)
        a = a.view(-1, k, q)
        b = b.view(-1, k, q)

        # calculate theta (rad)
        th = w * t + o
        y = a * torch.cos(th) + b * torch.sin(th)
        y = y.sum(dim=1)  # sum_k

        return y

    def calc_loss(self, pred_y: Tsr, tg: Tsr, **params) -> Tsr:
        y = tg[:, -1, :]
        loss = loss_quantile(pred_y, y)
        loss += self.calc_loss_train_constraint(**params)
        return loss

    def calc_loss_train_constraint(self, **params) -> Tsr:
        if "batch" not in params:  # possibly in prediction context
            return 0.0

        bti_org, __bti, btc, bkn, __btg = params["batch"]
        __x, a, b, w, o = self._forward(bti_org, btc, bkn)
        # l2 = nn.MSELoss()
        l1 = nn.SmoothL1Loss()
        loss_constraint = (
            l1(a[:-1], a[1:])
            + l1(b[:-1], b[1:])
            + l1(w[:-1], w[1:])
            + l1(o[:-1], o[1:])
        )
        return loss_constraint


class Trend(ModelBase):
    name = "trend"

    def __init__(
        self,
        dim_ins: tuple,
        dim_out: int,
        ws: int,
        dim_emb=3,
        n_heads=3,
        n_layers=1,
        k=5,  # the numbers of relu curves
    ):
        super().__init__(
            dim_ins,
            dim_out,
            ws,
            dim_emb,
            n_heads,
            n_layers,
            k,
        )  # after this call, to be enabled to access `self.args`

        # linears
        self.ak = nn.Linear(ws * dim_emb, k).double()
        self.bk = nn.Linear(ws * dim_emb, self.n_quantiles).double()
        self.ok = nn.Linear(ws * dim_emb, k).double()

        # activation
        self.relu = nn.ReLU()

        # constraint
        self.loss_constraint = 0

        # initialize weights
        nn.init.xavier_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.ok.weight)
        for fc in [self.ak, self.bk, self.ok]:
            nn.init.zeros_(fc.bias)

    def pretrain(self, ti: Tsr, tc: Tsr, kn: Tsr) -> Tsr:
        return self._pretrain(ti, tc, kn)

    def _forward(self, ti: Tsr, tc: Tsr, kn: Tsr):
        # embedding
        # - for X
        x = self.make_x(ti, tc, kn)  # as input
        emb_encode = self.emb_encode(x) * numpy.sqrt(x.shape[-1])
        emb_encode = emb_encode.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb_encode = self.pos(emb_encode)

        # transform
        def reshape(tsr: Tsr):
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.dim_emb * self.args.ws)

        h = reshape(self.tr.encoder(emb_encode))
        a = self.ak(h)
        b = self.bk(h)
        o = torch.sigmoid(self.ok(h))  # almost ti in (0, 1), by 2200-1-1

        return x, a, b, o

    def forward(self, ti: Tsr, tc: Tsr, kn: Tsr):
        x, a, b, o = self._forward(ti, tc, kn)

        # adjusting the shape
        k, q = self.args.k, self.n_quantiles
        kq = k * q
        dim_ti = self.args.dim_ins[0]
        _ti = x[:, :, :dim_ti]
        t = _ti[:, -1, :].repeat(1, kq).view(-1, k, q)  # shape = (B, k, q)
        a = a.view(-1, k, 1).repeat(1, 1, q)  # (B, k, q)
        b = b.view(-1, 1, q).repeat(1, k, 1)  # (B, k, q)
        o = o.view(-1, k, 1).repeat(1, 1, q)  # (B, k, q)

        # calculate trend line with t
        y = a * self.relu(t - o) + b
        y = y.sum(dim=1)  # ¥sum_k y_{b, k, q}
        return y

    def calc_loss(self, pred_y: Tsr, tg: Tsr, **params) -> Tsr:
        y = tg[:, -1, :]
        loss = loss_quantile(pred_y, y) + loss_mse(pred_y, y)
        loss += self.calc_loss_train_constraint(**params)
        return loss

    def calc_loss_train_constraint(self, **params) -> Tsr:
        if "batch" not in params:  # possibly in prediction context
            return 0.0

        bti_org, __bti, btc, bkn, __btg = params["batch"]
        __x, a, b, o = self._forward(bti_org, btc, bkn)
        # l2 = nn.MSELoss()
        # l1 = nn.L1Loss()
        l1 = nn.SmoothL1Loss()
        n_smooth_steps = 3
        loss_constraint = 0.0
        for idx in range(1, n_smooth_steps + 1):
            loss_constraint += (
                l1(a[:-idx], a[idx:]) + l1(b[:-1], b[1:]) + l1(o[:-1], o[1:])
            )
        # loss_constraint = l2(a[:-1], a[1:]) + l2(b[:-1], b[1:]) + l2(o[:-1], o[1:])
        return loss_constraint
