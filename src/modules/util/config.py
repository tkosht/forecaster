import yaml
import functools
from .items import Items


g_default_config_file = "conf/app.yml"


class Config(object):
    def __init__(self, config_file=g_default_config_file):
        self.config_file = config_file
        self.items = Items()
        self._load()

    def _load(self):
        with open(self.config_file, "r") as f:
            _config = yaml.safe_load(f)
            assert isinstance(_config, dict)
        self.items.setup(_config)

    def get(self, key: str):
        return self.items.get(key)


def load_params(params_file: str) -> callable:
    @functools.wraps(load_params)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = Config(params_file)
            kwargs.update(dict(params=cfg_params.items))
            return f(*args, **kwargs)

        return _wrapper

    return _decorator


if __name__ == "__main__":
    cfg = Config()
    print("get(target_keys):", cfg.get("target_keys"))
    print("cfg.items.~.target_keys:", cfg.items.data_spec.target_keys)
    assert id(cfg.items.data_spec.target_keys) == id(cfg.get("target_keys"))
    print("OK")

    @load_params(params_file="conf/ex.yml")
    def print_params(**kwargs):
        print(kwargs)

    print_params(a="1.2", b="xyz")
