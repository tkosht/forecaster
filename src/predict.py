import pandas
# import torch
# from pytorch_forecasting.metrics import MAE
# from pytorch_forecasting.models import TemporalFusionTransformer


if __name__ == "__main__":
    # from .modules.util.config import Config
    # cfg = Config()
    df = pandas.read_csv("data/stallion.csv", parse_dates=True, dtype={"month": str})

#     # load the best model according to the validation loss (given that
#     # we use early stopping, this is not necessarily the last epoch)
#     best_model_path = trainer.checkpoint_callback.best_model_path
#     print("best_model_path:", best_model_path)
#     best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
#     # calculate mean absolute error on validation set
#     actuals = torch.cat([y for x, y in iter(loader_validset)])
#     predictions = best_tft.predict(loader_validset)
#     MAE()(predictions, actuals)