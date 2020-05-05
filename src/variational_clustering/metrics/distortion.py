import numpy as np


__all__ = ["distortion"]


def distortion(errors):
    # return reduce(lambda x, y: x+y, errors)
    return np.mean(np.array(errors) ** 2)
    # return np.sum(np.array(errors))


if __name__ == "__main__":
    pass
