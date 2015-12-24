import time
import numpy as np
import cProfile


def grad(x,y):
	gra = 1/x - 1/y
	return np.sum(grad)

def grad2(x,y):
	gra = 1/y * (1 - x/y)
	return np.sum(grad)


x = np.random.rand(128) + 1
y = np.random.rand(128) + 1
start = time.time()
for i in range(100000):
	gra = 1/x - 1/y
end = time.time()

start2 = time.time()
for i in range(100000):
	gra = 1/y * (1 - x/y)
end2 = time.time()

print(end - start)
print(end2 - start2)