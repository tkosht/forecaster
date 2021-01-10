import pytest
from src.modules.util.config import Config, load_params


class TestConfig(object):
    @pytest.fixture
    def cfg(self) -> Config:
        config_file = "tests/modules/util/conf/app.yml"
        _cfg = Config(config_file)
        return _cfg

    def test_load(self, cfg: Config):
        assert cfg.items.model.name == "test_model"
        assert cfg.items.model.trainer.epoch_pretrain == 100
        assert cfg.items.model.trainer.epoch == 300
        assert cfg.items.model.optimizer.lr == 0.01
        assert cfg.items.model.loss.quantiles == [0.05, 0.30, 0.50, 0.70, 0.95]
        assert cfg.items.dataset.name == "custom"
        assert cfg.items.dataset.path == "data/data.tsv"

    def test_decorator(self, cfg: Config):
        @load_params("tests/modules/util/conf/params.yml")
        def _check_deco(a, b, **kwargs):
            params = kwargs["params"]
            assert a == 1.25
            assert b == "hello"
            assert params.n_encoder_layer == 3
            assert params.n_decoder_layer == 5
            assert params.n_heads == 4
            assert params.n_embedding == 16

        _check_deco(a=1.25, b="hello")
