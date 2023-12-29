from abc import ABC, abstractmethod


class callback(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def on_begin(self):
        pass

    @abstractmethod
    def on_iteration(self):
        pass


class callback_demo(callback):
    def __init__(self, params):
        self.params = params

    def on_begin(self):
        print("log on begin")

    def on_iteration(self):
        print("on_iteration")


def worker(callbacks):
    for func in callbacks:
        func.on_begin()
    for _ in range(50):
        for func in callbacks:
            func.on_iteration()
    print("worker ended")


if __name__ == "__main__":
    A = callback_demo("42")
    worker([A])
