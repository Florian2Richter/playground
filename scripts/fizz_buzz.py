# create a list of intergers,
n = 30

for entry in range(1, n + 1):
    if entry % 15 == 0:
        print("FizzBuzz!")
    elif entry % 5 == 0:
        print("Buzz!")
    elif entry % 3 == 0:
        print("Fizz!")
    else:
        print(entry)
