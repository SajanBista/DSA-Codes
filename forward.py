import math

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def newton_forward_interpolation(x, y, xp):
    n = len(x)
    h = x[1] - x[0]
    s = (xp - x[0]) / h

    # Create a forward difference table
    fd = y.copy()
    for i in range(1, n):
        for j in range(n - i):
            fd[j] = fd[j + 1] - fd[j]

    # Calculate the interpolated value
    v = fd[0]
    for i in range(1, n):
        p = 1
        for k in range(1, i + 1):
            p *= (s - k + 1)
        v += (p * fd[i]) / factorial(i)

    return v

if __name__ == "__main__":
    n = int(input("Enter the number of data points: "))
    x = []
    y = []
    for i in range(n):
        xi, yi = map(float, input(f"Enter the value of x and y at i={i}: ").split())
        x.append(xi)
        y.append(yi)

    xp = float(input("Enter the value at which interpolated value is needed: "))

    interpolated_value = newton_forward_interpolation(x, y, xp)
    print("Interpolated value = ", interpolated_value)