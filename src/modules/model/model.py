import inspect
import numpy
import torch
import torch.nn as nn
from ..util.items import Items

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


class Cyclic(M):
    def __init__(
        self,
        dim_ins: tuple,
        dim_out: int,
        ws: int,
        dim_emb=4 * 2,  # dim for embedding
        n_heads=4,
        k=5,  # the numbers of sin/cos curves
        n_layers=1,  # layers of multi-heads
        n_quantiles=7,
    ):
        super().__init__()  # after this call, to be enabled to access `self.args`
        n_dim = sum(dim_ins[1:])
        self.emb_X = nn.Linear(n_dim, dim_emb)  # .double()
        self.emb_y = nn.Linear(dim_out, dim_emb)  # .double()
        S = dim_ins[1]  # S: seq length
        max_len = max(16, S)
        self.pos = PositionalEncoding(d_model=dim_emb, max_len=max_len)

        # Transformers
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tr = nn.Transformer(**prm)  # .double()

        # linears
        self.dc = nn.Linear(ws * dim_emb, n_quantiles)  # .double()
        self.ak = nn.Linear(ws * dim_emb, k * n_quantiles)  # .double()
        self.bk = nn.Linear(ws * dim_emb, k * n_quantiles)  # .double()
        self.wk = nn.Linear(ws * dim_emb, k)  # .double()
        self.ok = nn.Linear(ws * dim_emb, k)  # .double()

        # initialize weights
        weight_interval = 0.1
        nn.init.uniform_(self.emb_X.weight, -weight_interval, weight_interval)
        nn.init.kaiming_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.ok.weight)

        for fc in [self.emb_X, self.ak, self.bk, self.wk, self.ok]:
            nn.init.zeros_(fc.bias)

    def pretrain(self, ti: Tsr, tc: Tsr, kn: Tsr, tg: Tsr):
        # concat input
        x = torch.cat([tc, kn], dim=-1)

        # make mask
        B, W, D = x.shape
        zr = torch.zeros((1, W, D))
        mask_vector = -1 * torch.ones_like(zr)  # (1, W, D)
        probs = 0.15 * torch.ones(1, W, 1)  # (1, W, 1)
        msk = torch.bernoulli(probs).repeat(1, 1, D)  # (1, W, D)
        mask_vector *= msk  # (1, W, D)
        mask_vector = mask_vector.repeat(B, 1, 1)
        unmsk = 1 - msk.type(torch.bool).type(torch.long)  # (1, W, D)
        assert (msk[0, :, 0] + unmsk[0, :, 0] == torch.ones(W)).all().item()

        # masking
        x *= unmsk.repeat(B, 1, 1).to(x.device)
        x += mask_vector.to(x.device)

        # embedding
        emb_X = self.emb_X(x) * numpy.sqrt(x.shape[-1])
        emb_X = emb_X.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb_X = self.pos(emb_X)

        # - for y
        y = tg
        emb_y = self.emb_y(y) * numpy.sqrt(y.shape[-1])
        emb_y = emb_y.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)

        # transform
        def reshape(tsr: Tsr) -> Tsr:
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.ws * self.args.dim_emb)

        encoded = reshape(self.tr(emb_X, emb_y))
        y = self.dc(encoded)
        return y

    def forward(self, ti: Tsr, tc: Tsr, kn: Tsr, tg: Tsr = None):
        # embedding
        # - for X
        x = torch.cat([tc, kn], dim=-1)
        emb_X = self.emb_X(x) * numpy.sqrt(x.shape[-1])
        emb_X = emb_X.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb_X = self.pos(emb_X)

        # transform
        def reshape(tsr: Tsr) -> Tsr:
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.ws * self.args.dim_emb)

        z = reshape(self.tr.encoder(emb_X))

        a = self.ak(z)
        b = self.bk(z)

        pi = numpy.pi
        w = torch.sigmoid(self.wk(z)) / (2 * pi)
        o = torch.sigmoid(self.ok(z)) / (2 * pi)
        # o = pi/2 + torch.sin(self.ok(z)) / (pi/2)      # map to (0, pi)
        # w = self.wk(z)
        # o = self.ok(z)

        # adjusting the shape
        k, q = self.args.k, self.args.n_quantiles
        kq = k * q
        t = ti[:, -1, :].repeat(1, kq).view(-1, k, q)
        w = w.view(-1, self.args.k, 1).repeat(1, 1, 7).view(-1, k, q)
        o = o.view(-1, self.args.k, 1).repeat(1, 1, 7).view(-1, k, q)
        a = a.view(-1, k, q)
        b = b.view(-1, k, q)

        # calculate theta (rad)
        th = 2 * pi * (w * t + o)
        y = a * torch.cos(th) + b * torch.sin(th)
        y = y.sum(dim=1)  # sum_k

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
        self.emb_X = nn.Linear(sum(dim_ins[0:]), dim_emb)  # .double()
        S = dim_ins[1]  # S: seq length
        max_len = max(16, S)
        self.pos = PositionalEncoding(d_model=dim_emb, max_len=max_len)

        # Transformers
        prm = dict(
            d_model=dim_emb,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
        )
        self.tra = nn.Transformer(**prm)  # .double()
        self.trb = nn.Transformer(**prm)  # .double()
        self.tro = nn.Transformer(**prm)  # .double()

        # linears
        self.ak = nn.Linear(ws * dim_emb, k)  # .double()
        self.bk = nn.Linear(ws * dim_emb, n_quantiles)  # .double()
        self.ok = nn.Linear(ws * dim_emb, k)  # .double()
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.ok.weight)

    def forward(self, ti: Tsr, tc: Tsr, kn: Tsr):
        # embedding
        x = torch.cat([ti, tc, kn], dim=-1)
        emb_X = self.emb_X(x) * numpy.sqrt(x.shape[-1])
        emb_X = emb_X.transpose(1, 0)  # (B, W, Demb) -> (W, B, Demb)
        emb_X = self.pos(emb_X)

        # transform
        def reshape(tsr: Tsr):
            _tsr = tsr.transpose(1, 0)  # (W, B, Demb) -> (B, W, Demb)
            return _tsr.reshape(-1, self.args.dim_emb * self.args.ws)

        ha = reshape(self.tra.encoder(emb_X))
        hb = reshape(self.trb.encoder(emb_X))
        ho = reshape(self.tro.encoder(emb_X))
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
