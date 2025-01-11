import math

def f(x):
    return 3*x**2 - 6*x + 2

def g(x):
    return math.sqrt((2*x - 2/3))

def fixed_point_iteration(x0, tol, max_iter):
    step = 1
    print("Step\tx0\tf(x0)\tx1\tf(x1)")

    while True:
        x1 = g(x0)
        f_x0 = f(x0)
        f_x1 = f(x1)
        print(f"{step}\t{x0:.6f}\t{f_x0:.6f}\t{x1:.6f}\t{f_x1:.6f}")

        if abs(x1 - x0) < tol:
            return x1

        if step > max_iter:
            print("Not convergent within the maximum number of iterations.")
            return None

        x0 = x1
        step += 1

x0 = float(input("Enter initial guess: "))
tol = float(input("Enter tolerable error: "))
max_iter = int(input("Enter maximum iterations: "))

root = fixed_point_iteration(x0, tol, max_iter)

if root is not None:
    print(f"Root is: {root:.6f}")