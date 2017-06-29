import timeit
def fun1(x,y):
    return x**2 + y**3
t_start = timeit.default_timer()
z= fun1(109.2, 367.1)
t_end = timeit.default_timer()
cost = t_end - t_start
print("Time cost of this function is %f" % cost)