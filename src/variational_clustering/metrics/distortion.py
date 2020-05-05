import numpy as np


__all__ = ["distortion"]


def distortion(errors):
    # return np.mean(np.array(errors))
    return np.sum(np.array(errors))


if __name__ == "__main__":
    pass
