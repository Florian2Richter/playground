import numpy as np

# create a list of intergers,
n = 30
int_list = np.arange(0, n)

for entry in int_list:
    if entry % 15 == 0:
        print("FizzBuzz!")
    elif entry % 5 == 0:
        print("Buzz!")
    elif entry % 3 == 0:
        print("Fizz!")
    else:
        print(entry)
