from .ptl import PTL

__factory = {
    'ptl': PTL,
}


def create_model(cfg):
    return __factory[cfg.name](cfg)
