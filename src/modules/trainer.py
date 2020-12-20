import datetime
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import pickle
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

from .util.items import Items
from .dataset.batcher import BatchMaker

from .model.model import Tsr


class Trainer(object):
    def __init__(self, model, optimizer, criterion, params: Items):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.params = params
        self.loss_train = None
        self.loss_valid = None

        self.train_model = model  # switch model if pretrain or train

        # setup tensorboard writer
        experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        logdir = f"result/{experiment_id}"
        writer = SummaryWriter(log_dir=logdir)
        self.writer = writer
        self.experiment_id = experiment_id

    def write_graph(self, trainset):
        self.writer.add_graph(self.model, (trainset.ti, trainset.tc, trainset.kn))

    def get_quantile(self, x: Tsr, alpha: float):
        idx = (numpy.array(self.params.quantiles) == alpha).argmax()
        return x[:, idx, :][..., 0]  # just to get a first dim

    def do_pretrain(self, dataset, epochs: int = 500):
        self.train_model = self.model.pretrain
        self._do_train(dataset, epochs)

    def do_train(self, dataset, epochs: int = 500):
        self.train_model = self.model
        self._do_train(dataset, epochs)

    def _get_train_mode(self):
        mode = "train" if self.train_model == self.model else "pretrain"
        return mode

    def _do_train(self, dataset, epochs: int = 500):
        train_mode = self._get_train_mode()
        print(f"##### {train_mode} #####")

        ti, tc, kn, tg = (
            dataset.trainset.ti,
            dataset.trainset.tc,
            dataset.trainset.kn,
            dataset.trainset.tg,
        )
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
                    y_pred = self.train_model(bti, btc, bkn)
                    loss = self.criterion(y_pred, btg[:, -1, :], self.params.quantiles)
                    losses.append(loss.item())
                    assert len(losses) == n_steps + 1

                    self.loss_train = loss.item()  # keep latest loss
                    k = self.params.log_interval
                    _loss = loss.item() if n_steps < k else numpy.mean(losses[-k:])
                    self.writer.add_scalar(
                        f"{train_mode}/loss/step/train", _loss, n_steps
                    )

                    # logging progress
                    if idx % self.params.log_interval == 0 and idx > 0 and bdx == 0:
                        mean_loss = numpy.mean(losses[-k:])
                        print(
                            f"{train_mode}/loss[{idx:03d}][{bdx:03d}][{n_steps:05d}]",
                            mean_loss,
                        )

                        # prediction with trainset
                        loss_train = self._predict(
                            idx, ti, tc, kn, tg, pred_mode="train"
                        )
                        self.loss_train = loss_train.item()

                        # prediction with testset
                        testset = dataset.create_testset()
                        loss_valid = self._predict(
                            idx,
                            testset.ti,
                            testset.tc,
                            testset.kn,
                            testset.tg,
                            pred_mode="valid",
                        )
                        self.loss_valid = loss_valid.item()

                    loss.backward()
                    return loss

                self.optimizer.step(closure)

            if idx % self.params.save_interval == 0 and idx > 0:
                mlflow.pytorch.log_model(
                    self.model, f"models.{idx:05d}", pickle_module=pickle
                )

    def _predict(
        self, idx, ti: Tsr, tc: Tsr, kn: Tsr, tg: Tsr, pred_mode="train"
    ) -> Tsr:
        batch = BatchMaker(bsz=self.params.batch_size)
        bti, btc, bkn, btg = (
            next(batch(ti)),
            next(batch(tc)),
            next(batch(kn)),
            next(batch(tg)),
        )
        with torch.no_grad():
            pred = self.model(bti, btc, bkn)
            loss = self.criterion(pred, btg[:, -1, :], self.params.quantiles)
        preds = self._make_predictions(pred, btg)
        self._write_log2tb(idx, preds, loss, pred_mode)
        return loss

    def _make_predictions(self, y_pred, tg) -> Tuple[Tsr, Tsr, Tsr, Tsr]:
        pred = y_pred.view(-1, len(self.params.quantiles), self.model.args.dim_out)
        p = self.get_quantile(pred, alpha=0.5)
        p10 = self.get_quantile(pred, alpha=0.1)
        p90 = self.get_quantile(pred, alpha=0.9)
        t = tg[:, -1, :][..., 0]
        return p, p10, p90, t

    def _write_log2tb(self, idx, preds, loss, pred_type="train") -> None:
        train_mode = self._get_train_mode()
        for n, (y0, yL, yH, t0) in enumerate(zip(*preds)):
            dct_pred = dict(p=y0, p10=yL, p90=yH, t=t0)
            self.writer.add_scalars(
                f"{train_mode}/prediction/epoch_{idx:03d}/{pred_type}",
                dct_pred,
                n,
            )
        self.writer.add_scalar(
            f"{train_mode}/loss/interval/{pred_type}", loss.item(), idx
        )

    def finalize(self, args):
        self._log_experiments(args)

    def _log_experiments(self, args):
        # experiment log
        hparams = dict(
            experiment_id=self.experiment_id,
            model=args.model,
            max_epoch=args.max_epoch,
            optimizer=str(self.optimizer),
            criterion=str(self.criterion),
        )
        self.writer.add_hparams(
            hparams,
            {
                "hparam/loss/train": self.loss_train,
                "hparam/loss/valid": self.loss_valid,
            },
        )
        self.writer.close()

        mlflow.pytorch.log_model(
            self.model,
            "models",
            registered_model_name="latest_model",
            pickle_module=pickle,
        )


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
    from .model.model import Cyclic, Trend
    from .dataset.dateset import DatesetToy

    args = get_args()

    # setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # create toy dataset
    B, W, Dout = 64, 14, 1
    toydataset = DatesetToy(Dout, W, args.model, device=device)
    trainset = toydataset.create_trainset()

    # setup criterion
    criterionner = dict(cyclic=quantile_loss, trend=quantile_loss_with_mse)
    criterion = criterionner[args.model]

    if args.resume:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        (rm,) = client.list_registered_models()
        lm = dict(rm)["latest_versions"][0]
        uri = f"models:/latest_model/{lm.version}"
        print(f"loading the latest model ... [{uri}]")
        model = mlflow.pytorch.load_model(uri)
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
            n_quantiles=len(criterion.__defaults__[0]),  # like len(quantiles)
        )
    model.to(device)
    print("model.args:", model.args)

    # setup optimizer and criterion
    # optimizer = optim.LBFGS(model.parameters(), lr=0.8)     # Newton
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    params = Items(is_readonly=True).setup(
        dict(
            batch_size=B,
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        )
    )
    trainer = Trainer(model, optimizer, criterion, params)

    # pretrain
    trainer.do_pretrain(toydataset, epochs=args.max_epoch)

    # train
    trainer.do_train(toydataset, epochs=args.max_epoch)

    # finalize
    trainer.finalize(args)
