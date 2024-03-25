def flo_generator(x):
    for item in x:
        yield item

generator = flo_generator([1,2,3,4,5,6]
              )

print(generator)
print(next(generator))
