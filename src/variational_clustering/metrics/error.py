__all__ = ["errors"]


def errors(faces, proxy):
    return [face.get_error(proxy) for face in faces]


if __name__ == "__main__":
    pass
