import pandas
from typing import Tuple
from .modules.util.config import Config
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer


class DatasetSplitter(object):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def create_data_spec(self) -> dict:
        cfg = self.cfg
        data_spec = dict(
            time_idx=cfg.get("time_index"),
            target=cfg.get("target"),
            group_ids=cfg.get("target_keys"),
            target_normalizer=GroupNormalizer(
                groups=cfg.get("target_keys"), coerce_positive=1.0
            ),  # use softplus with beta=1.0 and normalize by group
            static_categoricals=cfg.get("static").categorical,
            static_reals=cfg.get("static").numerical,
            time_varying_known_categoricals=cfg.get("known").categorical,
            variable_groups=cfg.get("variable_groups"),
            time_varying_known_reals=cfg.get("known").numerical,
            time_varying_unknown_categoricals=cfg.get("unknown").categorical,
            time_varying_unknown_reals=cfg.get("unknown").numerical,
        )
        return data_spec

    def create_prediction_spec(self) -> dict:
        max_prediction_length = cfg.get("max_prediction_length")
        max_encoder_length = cfg.get("max_encoder_length")

        prediction_spec = dict(
            min_encoder_length=0,  # allow predictions without history
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
        )
        return prediction_spec

    def create_dataset(self, df: pandas.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        data_spec = self.create_data_spec()

        preprocess_spec = dict(
            add_relative_time_idx=True,  # add as feature
            add_target_scales=True,  # add as feature
            add_encoder_length=True,  # add as feature
        )

        prediction_spec = self.create_prediction_spec()

        time_index_col = cfg.get("time_index")
        training_cutoff = df[time_index_col].max() - self.cfg.get("max_prediction_length")
        trainset = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            **data_spec,
            **preprocess_spec,
            **prediction_spec,
        )
        # create validation set (predict=True) which means to predict the
        # last max_prediction_length points in time for each series
        validset = TimeSeriesDataSet.from_dataset(
            trainset, df, predict=True, stop_randomization=True
        )
        return trainset, validset


def run_train(cfg, trainset, validset) -> None:
    import multiprocessing
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_forecasting.models import TemporalFusionTransformer

    # stop training, when loss metric does not improve on validation set
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )
    lr_monitor = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("result/lightning_logs")  # log to tensorboard
    # create trainer
    params = cfg.get("trainer").params
    trainer = pl.Trainer(
        callbacks=[lr_monitor, early_stop_callback],
        logger=logger,
        **params,
    )

    # initialise model
    params = cfg.get("model").params
    tft = TemporalFusionTransformer.from_dataset(
        trainset,
        loss=QuantileLoss(),
        **params,
    )
    print(tft.size())   # 29.6k parameters in model

    n_cores = multiprocessing.cpu_count()
    loader_trainset = trainset.to_dataloader(
        train=True, batch_size=cfg.get("trainset").batch_size, num_workers=n_cores
    )
    loader_validset = validset.to_dataloader(
        train=False, batch_size=cfg.get("validset").batch_size, num_workers=n_cores
    )

    # fit network
    trainer.fit(
        tft,
        train_dataloader=loader_trainset,
        val_dataloaders=loader_validset,
    )

    return


if __name__ == "__main__":
    cfg = Config()
    df = pandas.read_csv("data/stallion.csv", parse_dates=True, dtype={"month": str})

    splitter = DatasetSplitter(cfg)
    trainset, validset = splitter.create_dataset(df)
    run_train(cfg, trainset, validset)
