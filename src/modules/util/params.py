import functools
from omegaconf import OmegaConf


def params(params_file: str) -> callable:
    @functools.wraps(params)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = OmegaConf.load(params_file)
            kwargs.update(dict(params=cfg_params))
            return f(*args, **kwargs)

        return _wrapper

    return _decorator


def args(params_file: str) -> callable:
    @functools.wraps(params)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = OmegaConf.load(params_file)
            kwargs.update(**cfg_params)
            return f(*args, **kwargs)

        return _wrapper

    return _decorator
