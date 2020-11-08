class Item(dict):
    def __init__(self, is_readonly=False):
        self.is_readonly = is_readonly

    def setup(self, d: dict):
        self.update(d)
        _setup(self, d)
        return self

    def __str__(self):
        return str(vars(self))

    def __setattr__(self, name, value):
        if getattr(self, "is_readonly", False):
            if name in self.__dict__:
                raise self.ConstError(f"Can't rebind const ({name})")
        self.__dict__[name] = value


class Items(dict):
    def __init__(self, is_readonly=True):
        self.is_readonly = is_readonly

    def setup(self, d: dict):
        _setup(self, d, self.is_readonly)
        return self

    def get(self, key: str):
        return _get(self, key)


# set the member name as key
def _setup(obj, d: dict, is_readonly: bool):
    obj.update(d)
    for k, v in d.items():
        if isinstance(v, dict):
            itm = Item(is_readonly)
            _setup(itm, v, is_readonly)
            setattr(obj, k, itm)
            continue
        setattr(obj, k, v)


# get the first member values which name matches key
def _get(obj, key: str):
    for k, v in vars(obj).items():
        if k == key:
            return v
        if isinstance(v, Item):
            obj = _get(v, key)
            if obj is not None:
                return obj
    return None


if __name__ == '__main__':
    d = dict(
        first=dict(
            model="MLP",
            layers=[16, 32, 64, 32],
            lr=1e-3,
            act="ReLU",
        ),
        second=dict(
            model="VAE",
            x_dim=768,
            encoder=[512, 256, 128],
            latent_dim=64,
            decoder=[128, 256, 512],
            lr=1e-4,
            act="tanh",
        ),
        common=dict(
            loss="cross_entropy",
            optimizer="Adam"
        )
    )

    # test for accesibility
    itm = Items()
    itm.setup(d)

    # # test access member variables
    # ## like dic or member variable
    assert itm["first"]["lr"] == d["first"]["lr"]
    assert itm.first.lr == d["first"]["lr"]
    assert itm.get("first").lr == d["first"]["lr"]
    assert itm.get("first")["lr"] == d["first"]["lr"]
    assert itm.get("lr") == d["first"]["lr"]

    assert itm.second == itm.get("second")
    assert itm["second"] == itm.get("second")
    assert id(itm.second) == id(itm.get("second"))

    # ## using dict parameters
    def _call_function(loss: str, optimizer: str) -> None:
        assert loss == "cross_entropy"
        assert optimizer == "Adam"

    cmn = itm.get("common")
    _call_function(**cmn)

    # test for readonly
    itm = Items(readonly=True)
    itm.setup(d)
    try:
        itm.second = {"second": "abc"}
        assert False
    except TypeError:
        assert True

    print("OK")
