import numpy as np

points = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

# calculate euclidean distance for each of the points using numpy
#make an array
points_array = np.array([[item[0],item[1]] for item in points])
print(points_array.shape)

#calculate distance using broadcasting... something like a 3rd dimension with shift
A = points_array.reshape(5,1,2)
B = points_array.reshape(1,5,2)
diff = A - B
print(diff)
sq_diff = diff**2
distances = np.sqrt(np.sum(sq_diff, axis = -1))
print(distances)
