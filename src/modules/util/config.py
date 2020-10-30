import yaml
from .items import Items


class Config(object):
    def __init__(self, config_file="conf/prediction.yml"):
        self.config_file = config_file
        self.items = Items()
        self._load()

    def _load(self):
        with open(self.config_file, 'r') as f:
            _config = yaml.load(f)
        self.items.setup(_config)

    def get(self, key: str):
        return self.items.get(key)


if __name__ == "__main__":
    cfg = Config()
    print("target_keys:", cfg.get("target_keys"))
