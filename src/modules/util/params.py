import functools
from omegaconf import OmegaConf


def add_params(params_file: str) -> callable:
    @functools.wraps(add_params)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = OmegaConf.load(params_file)
            kwargs.update(dict(params=cfg_params))
            return f(*args, **kwargs)

        return _wrapper

    return _decorator


def add_args(params_file: str, as_default: bool = False) -> callable:
    @functools.wraps(add_args)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = OmegaConf.load(params_file)
            if as_default:
                cfg_params.update(kwargs)
                kwargs = cfg_params
            else:
                kwargs.update(cfg_params)
            return f(*args, **kwargs)

        return _wrapper

    return _decorator
