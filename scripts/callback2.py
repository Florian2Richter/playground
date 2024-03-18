from abc import ABC, abstractmethod


class Callback(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def on_begin():
        pass

    @abstractmethod
    def on_iteration():
        pass


class CallbackFlorian(Callback):
    def __init__(self):
        pass

    def on_begin(self):
        print("on_begin")

    def on_iteration(self):
        print("on_iteration")


def worker(func_list):
    for callback in func_list:
        callback.on_begin()

    for iteration in range(0, 50):
        # do something here
        for callback in func_list:
            callback.on_iteration()
    pass


if __name__ == "__main__":
    callback_flo = CallbackFlorian()
    worker([callback_flo])
