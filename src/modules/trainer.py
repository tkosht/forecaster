import datetime
import numpy
import torch
import torch.optim as optim

import mlflow
import pickle
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

from .util.items import Items
from .dataset.batcher import BatchMaker
from .dataset.dateset import Tsr, DateTensors


class Trainer(object):
    def __init__(self, model, optimizer, params: Items):
        self.model = model
        self.optimizer = optimizer
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
        d = next(trainset(self.params.batch_size))
        data_in = d.safe_tuple()[:-1]
        self.writer.add_graph(self.model, data_in)

    def get_quantile(self, x: Tsr, alpha: float):
        idx = (numpy.array(self.params.quantiles) == alpha).argmax()
        return x[:, idx, :][..., 0]  # just to get a first dim

    def do_pretrain(self, dataset, epochs: int = 500):
        self.train_model = self.model.pretrain
        self._do_train(dataset, epochs)
        self._save_model(model_title=f"pretrained_model_{self.model.name}")

    def do_train(self, dataset, epochs: int = 500):
        self.train_model = self.model
        self._do_train(dataset, epochs)
        self._save_model(model_title="latest_model")

    def _get_train_mode(self):
        mode = "train" if self.train_model == self.model else "pretrain"
        return mode

    def _do_train(self, dataset, epochs: int = 500):
        train_mode = self._get_train_mode()
        print(f"##### {train_mode} #####")

        # ti, tc, kn, tg = (
        #     dataset.trainset.ti,
        #     dataset.trainset.tc,
        #     dataset.trainset.kn,
        #     dataset.trainset.tg,
        # )
        # batch = BatchMaker(bsz=self.params.batch_size)

        # train loop
        n_steps = -1
        losses = []
        # # epoch loop
        for idx in range(epochs):
            # shuffle = numpy.random.permutation(range(len(ti)))
            # _ti, _tc, _kn, _tg = ti[shuffle], tc[shuffle], kn[shuffle], tg[shuffle]
            shuffled = dataset.trainset.create_shuffle()

            bsz = self.params.batch_size
            # # batch loop
            for bdx, (borig, bshuf) in enumerate(
                zip(dataset.trainset(bsz), shuffled(bsz))
            ):
                (bti, btv, btc, bkn, btg) = bshuf.safe_tuple()
                n_steps += 1

                def closure():
                    model_params = dict(batch_ordered=borig)
                    self.optimizer.zero_grad()
                    if train_mode == "pretrain":
                        loss = self.model.loss_pretrain(
                            bti, btv, btc, bkn, **model_params
                        )
                    else:
                        loss = self.model.loss_train(
                            bti, btv, btc, bkn, btg, **model_params
                        )
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
                        d = toydataset.trainset
                        loss_train = self._predict(
                            idx, d.ti, d.tc, d.kn, d.tg, pred_mode="train"
                        )
                        self.loss_train = loss_train.item()

                        # prediction with testset
                        d = dataset.create_testset()
                        loss_valid = self._predict(
                            idx, d.ti, d.tc, d.kn, d.tg, pred_mode="valid"
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
            btv = Tsr([]).to(bti.device)  # TODO: refactored
            pred = self.model(bti, btv, btc, bkn)
            loss = self.model.calc_loss(pred, btg)
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
        )
        self.writer.add_hparams(
            hparams,
            {
                "hparam/loss/train": self.loss_train,
                "hparam/loss/valid": self.loss_valid,
            },
        )
        self.writer.close()

    def _save_model(self, model_title="latest_model"):
        mlflow.pytorch.log_model(
            self.model,
            "models",
            registered_model_name=model_title,
            pickle_module=pickle,
        )


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
        "--max-epoch-pretrain",
        type=int,
        default=300,
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # create toy dataset
    B, W, Dout = 64, 14, 1
    toydataset = DatesetToy(Dout, W, args.model, device=device)
    trainset = toydataset.create_trainset()

    if args.resume:

        def search_latest_model_info():
            from mlflow.tracking import MlflowClient

            mlf_client = MlflowClient()
            for m in mlf_client.list_registered_models():
                if m.name == "latest_model":
                    return m
            return None

        reg_info = search_latest_model_info()
        assert reg_info is not None
        info_latest = dict(reg_info)["latest_versions"][0]
        uri = f"models:/latest_model/{info_latest.version}"
        print(f"loading the latest model ... [{uri}]")
        model = mlflow.pytorch.load_model(uri)
        print("loading done.")

        # recreate toydataset corresponding to the model
        args.model = model.name
        toydataset = DatesetToy(Dout, W, args.model, device=device)
        trainset = toydataset.create_trainset()
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
            n_layers=1,
            k=5,
        )
    model.to(device)
    print("model.args:", model.name, model.args)

    # setup optimizer
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
    trainer = Trainer(model, optimizer, params)

    # graph to tensorboard
    trainer.write_graph(trainset)

    # pretrain
    trainer.do_pretrain(toydataset, epochs=args.max_epoch_pretrain)

    # train
    trainer.do_train(toydataset, epochs=args.max_epoch)

    # finalize
    trainer.finalize(args)
