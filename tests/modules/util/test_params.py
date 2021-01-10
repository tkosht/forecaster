from src.modules.util.params import params, args


class TestParams(object):
    def test_decorator(self):
        @params("tests/modules/util/conf/params.yml")
        def _check_deco(a, b, params):
            assert a == 1.25
            assert b == "hello"
            assert params.n_encoder_layer == 3
            assert params.n_decoder_layer == 5
            assert params.n_heads == 4
            assert params.n_embedding == 16

        _check_deco(a=1.25, b="hello")

    def test_args(self):
        @args("tests/modules/util/conf/params.yml")
        def _check_deco(a, b, n_encoder_layer, n_decoder_layer, n_heads, n_embedding):
            assert a == 0.25
            assert b == "world"
            assert n_encoder_layer == 3
            assert n_decoder_layer == 5
            assert n_heads == 4
            assert n_embedding == 16

        _check_deco(a=0.25, b="world")

    def test_json(self):
        @params("tests/modules/util/conf/app.json")
        def _check_deco(a, b, params):
            assert a == -0.25
            assert b == "!!!"
            assert params.model.name == "test_model"
            assert params.model.trainer.epoch_pretrain == 100
            assert params.model.trainer.epoch == 300
            assert params.model.optimizer.lr == 0.01
            assert params.model.loss.quantiles == [0.05, 0.30, 0.50, 0.70, 0.95]
            assert params.dataset.name == "custom"
            assert params.dataset.path == "data/data.tsv"

        _check_deco(a=-0.25, b="!!!")
