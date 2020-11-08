import inspect
import numpy
import torch
import torch.nn as nn
from ..util.items import Items


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.set_params()

    def set_params(self):
        _frame = inspect.currentframe()
        _locals = _frame.f_back.f_back.f_locals
        assert _frame.f_back.f_back.f_locals["__class__"] == type(self)
        params = {k : v for k, v in _locals.items() if k not in ["self"] and k[:1] != "_"}
        self.params = Items().setup(params)


class Model(M):
    def __init__(self, dim=16, n_heads=4, ws=8, n_quantiles=7):
        super().__init__()      # after this call, to be enabled to access `self.params`
        self.cyclic = Cyclic(dim=dim, n_heads=4, n_quantiles=self.params.n_quantiles)
        self.trend = None
        self.recent = None
        self.ws = ws        # window size

    def forward(self, ti: torch.Tensor, tc: torch.Tensor = None, kn: torch.Tensor = None, un: torch.Tensor = None):
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
        cyclic = self.cyclic(ti, tc, kn)
        trend = self.trend(ti, tc, kn)
        ws = self.ws
        recent = self.recent(ti, kn, tc[:, -ws:, :], un[:, -ws:, :])
        y = cyclic + trend + recent
        return y


def create_linears(dim_emb, kq, ws):
    return nn.Seuquential(nn.Linear(dim_emb*ws, dim_emb), nn.ReLU(), nn.Linear(dim_emb, kq))


class Cyclic(M):
    def __init__(self, dim_ins: tuple, dim_out: int, ws: int, dim_emb=5, n_heads=4, k=3, n_quantiles=7, n_layers=2):
        super().__init__()      # after this call, to be enabled to access `self.params`
        self.emx = nn.Linear(sum(dim_ins[1:]), dim_emb)
        self.emy = nn.Linear(1, dim_emb)

        # Transformers
        prm = dict(d_model=dim_emb, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers)
        self.tra = nn.Transformer(**prm)
        self.trb = nn.Transformer(**prm)
        self.trw = nn.Transformer(**prm)
        self.tro = nn.Transformer(**prm)

        # linears
        self.ak = nn.Linear(dim_emb*ws, k*n_quantiles)
        self.bk = nn.Linear(dim_emb*ws, k*n_quantiles)
        self.wk = nn.Linear(dim_emb*ws, k)
        self.ok = nn.Linear(dim_emb*ws, k)

        # initialize weights
        nn.init.kaiming_normal_(self.ak.weight)
        nn.init.kaiming_normal_(self.bk.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.ok.weight)

    def forward(self, ti: torch.Tensor, tc: torch.Tensor, kn: torch.Tensor, tg: torch.Tensor):
        # embedding
        x = torch.cat([tc, kn], dim=-1)
        emx = self.emx(x)
        # emy = self.emy(tg)

        # transform
        # h = self.tr(emx, emy)
        def reshape(tsr: torch.Tensor):
            return tsr.view(-1, self.params.dim_emb * self.params.ws)

        ha = reshape(self.tra.encoder(emx))
        hb = reshape(self.trb.encoder(emx))
        hw = reshape(self.trw.encoder(emx))
        ho = reshape(self.tro.encoder(emx))
        a = self.ak(ha)
        b = self.bk(hb)
        w = torch.sigmoid(self.wk(hw)) / (2 * numpy.pi)
        o = torch.sigmoid(self.ok(ho)) / numpy.pi

        # adjusting the shape
        kq = self.params.k * self.params.n_quantiles
        t = ti[:, -1, :].repeat(1, kq)
        w = w.view(-1, self.params.k, 1).repeat(1, 1, 7).view(-1, kq)
        o = o.view(-1, self.params.k, 1).repeat(1, 1, 7).view(-1, kq)

        # calculate theta (rad)
        th = 2 * numpy.pi * w * t + o
        y = a * torch.cos(th) + b * torch.sin(th)
        y = y.view(-1, self.params.k, self.params.n_quantiles)
        y = y.sum(dim=1)

        return y


def _create_toydataset(B, W, Dout):
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
    t0 = torch.arange(0, W).view(1, W, Dout)
    ti = t0
    for idx in range(B-1):
        t1 = t0 + idx + 1
        ti = torch.cat([ti, t1], axis=0)
    ti = ti.float()

    tc = torch.rand((B, W, 3))  # shape: (B, W, Dtc)
    kn = torch.rand((B, W, 4))  # shape: (B, W, Dkn)
    tg = torch.sin(ti).repeat(1, 1, Dout)  # target
    return ti, tc, kn, tg


if __name__ == "__main__":
    import torch.optim as optim

    # create toy dataset
    B, W, Dout = 64, 4, 1

    ti, tc, kn, tg = _create_toydataset(B, W, Dout)

    # setup model
    dims = (ti.shape[-1], tc.shape[-1], kn.shape[-1])
    model = Cyclic(dim_ins=dims, dim_out=tg.shape[-1], ws=ti.shape[1], dim_emb=4*2, n_heads=2, k=3, n_quantiles=7)

    # setup optimizer and criterion
    # optimizer = optim.LBFGS(model.parameters(), lr=0.8)     # Newton
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)

    def quantile_loss(pred_y, tg, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        losses = []
        for idx, qtl in enumerate(quantiles):
            err = tg - pred_y[..., idx].unsqueeze(-1)       # (B, 1)
            losses.append(torch.max((qtl - 1) * err, qtl * err).unsqueeze(-1))  # (B, 1, 1)
        losses = torch.cat(losses, dim=2)
        loss = losses.sum(dim=(1, 2)).mean()
        # loss += nn.MSELoss()(pred_y, tg.repeat(1, len(quantiles)))
        return loss

    criterion = quantile_loss
    cri_params = Items().setup(dict(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

    N = 10 * 1000
    batch_losses = []
    losses = []
    for idx in range(N):
        shuffle = numpy.random.permutation(range(len(ti)))
        t = ti[shuffle]

        def closure():
            optimizer.zero_grad()
            y_pred = model(t, tc, kn, tg)
            loss = criterion(y_pred, tg[:, -1, :], **cri_params)
            batch_losses.append(loss.item())
            if idx % 20 == 0 and idx > 0:
                mean_loss = numpy.array(batch_losses[idx-20: idx]).mean()
                print(f"loss[{idx:03d}]", mean_loss)
                losses.append(mean_loss)
            loss.backward()
            return loss
        optimizer.step(closure)

    import pandas
    from matplotlib import pyplot

    df = pandas.DataFrame({"loss": losses})
    df.plot(figsize=(10, 5))
    pyplot.savefig("img/loss.png")

    # predict for trainset
    def _get_quantile(x: torch.Tensor, alpha: float):
        idx = (numpy.array(cri_params.quantiles) == alpha).argmax()
        return x[:, idx, :][..., 0]     # just to get a first dim

    def do_predict(ti: torch.Tensor, tc: torch.Tensor, kn: torch.Tensor, tg: torch.Tensor, name="trainset"):
        with torch.no_grad():
            y_pred = model(ti, tc, kn, tg)

            # arrange outputs
            y_pred = y_pred.view(-1, len(cri_params.quantiles), Dout)
            p = _get_quantile(y_pred, alpha=.5)
            p10 = _get_quantile(y_pred, alpha=.1)
            p90 = _get_quantile(y_pred, alpha=.9)
            t = tg[:, -1, :][..., 0]

            # plot and save
            df = pandas.DataFrame({"p": p, "t": t})
            df.plot(figsize=(10, 5))
            pyplot.fill_between(df.index, p10, p90, facecolor='b', alpha=0.2)
            pyplot.savefig(f"img/pred_{name}.png")
        return

    do_predict(ti, tc, kn, tg, "trainset")

    # predict for testset
    offset = 10
    sz = 16
    test_ti = (ti + len(ti))[offset:offset+sz]
    test_tc = tc[offset:offset+sz]
    test_kn = kn[offset:offset+sz]
    test_tg = tg[offset:offset+sz]

    do_predict(test_ti, test_tc, test_kn, test_tg, "testset")
