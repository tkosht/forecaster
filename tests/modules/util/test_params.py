from src.modules.util.params import add_params, add_args


class TestParams(object):
    def test_decorator(self):
        @add_params("tests/modules/util/conf/params.yml")
        def _check_deco(a, b, params):
            assert a == 1.25
            assert b == "hello"
            assert params.n_encoder_layer == 3
            assert params.n_decoder_layer == 5
            assert params.n_heads == 4
            assert params.n_embedding == 16

        _check_deco(a=1.25, b="hello")

    def test_args(self):
        @add_args("tests/modules/util/conf/params.yml")
        def _check_deco(a, b, n_encoder_layer, n_decoder_layer, n_heads, n_embedding):
            assert a == 0.25
            assert b == "world"
            assert n_encoder_layer == 3
            assert n_decoder_layer == 5
            assert n_heads == 4
            assert n_embedding == 16

        _check_deco(a=0.25, b="world")

    def test_args_overwrite(self):
        @add_args("tests/modules/util/conf/params.yml")
        def _check_deco(n_encoder_layer, n_decoder_layer, n_heads, n_embedding):
            assert n_encoder_layer == 3
            assert n_decoder_layer == 5
            assert n_heads == 4
            assert n_embedding == 16

        # check if config values are available, i.e. ignore specified args
        _check_deco(n_encoder_layer=1, n_decoder_layer=7)

    def test_args_as_default_false(self):
        @add_args("tests/modules/util/conf/params.yml", as_default=False)
        def _check_deco(n_encoder_layer, n_decoder_layer, n_heads, n_embedding):
            assert n_encoder_layer == 3
            assert n_decoder_layer == 5
            assert n_heads == 4
            assert n_embedding == 16

        # check if config values are available, i.e. ignore specified args
        _check_deco(n_encoder_layer=1, n_decoder_layer=7)

    def test_args_as_default_true(self):
        @add_args("tests/modules/util/conf/params.yml", as_default=True)
        def _check_deco(n_encoder_layer, n_decoder_layer, n_heads, n_embedding):
            assert n_encoder_layer == 1
            assert n_decoder_layer == 7
            assert n_heads == 4
            assert n_embedding == 16

        # check if config values are available, i.e. ignore specified args
        _check_deco(n_encoder_layer=1, n_decoder_layer=7)

    def test_json(self):
        @add_params("tests/modules/util/conf/app.json")
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
