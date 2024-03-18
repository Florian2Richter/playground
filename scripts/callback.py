from abc import ABC, abstractmethod


class callback(ABC):

    def __init__(self):

        pass

    @abstractmethod  # declaration as abstract class
    def on_begin(self):

        pass

    @abstractmethod
    def on_iteration(self):

        pass


class callback_florian(callback):

    def __init__(self, params):

        self.params = params

    def on_begin(self):

        print("on_begin")

    def on_iteration(self):

        print("on_iteration")


##### Callback functions


def worker(func_list):

    timestamp = 0

    for callback in func_list:

        callback.on_begin()

    for timestamp in range(0, 50):

        # do something unrelated here

        for callback in func_list:

            callback.on_iteration()


def callback_function(param1):  # der callback haengt vom worker ab!

    return "string" + param1


if __name__ == "__main__":

    A = callback_florian(42)

    worker([A])
