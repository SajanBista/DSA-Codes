import math

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def newton_backward_interpolation(x, y, xp):
    n = len(x)
    h = x[1] - x[0]
    s = (xp - x[n-1]) / h

    # Create a backward difference table
    bd = y.copy()
    for i in range(1, n):
        for j in range(n - i, 0, -1):
            bd[j] = bd[j] - bd[j - 1]

    # Calculate the interpolated value
    v = bd[n - 1]
    for i in range(1, n):
        p = 1
        for k in range(1, i + 1):
            p *= (s + k - 1)
        v += (p * bd[n - i - 1]) / factorial(i)

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

    interpolated_value = newton_backward_interpolation(x, y, xp)
    print("Interpolated value = ", interpolated_value)