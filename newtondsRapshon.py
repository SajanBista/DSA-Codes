import math

def f(x):
    return x**3 - x - 1

def df(x):
    return 3*x**2 - 1

def newton_raphson(x0, tol, max_iter):
    for i in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        if abs(x1 - x0) < tol:
            return x1, i+1  # Return the root and number of iterations
        x0 = x1
    return None, max_iter  # Return None and max_iter if not converged

if __name__ == "__main__":
    x0 = float(input("Enter initial guess: "))
    tol = float(input("Enter tolerance: "))
    max_iter = 100

    root, iterations = newton_raphson(x0, tol, max_iter)

    if root is not None:
        print(f"Root found after {iterations} iterations: {root}")
    else:
        print("Solution not found within specified iterations.")