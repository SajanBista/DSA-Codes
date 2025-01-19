numerical method some problem solved

"""import math

# Define the function
def f(x):
    return 3 * (x ** 2) + 2 * x - 5

# Main function
def trapezoidal_composite_rule():
    print("Sajan Bista Trapezoidal Composite Rule")

    # Input lower and upper bounds
    x0 = float(input("Enter the lower bound: "))
    xn = float(input("Enter the upper bound: "))

    # Input number of segments
    k = int(input("Enter the number of segments: "))

    # Calculate step size
    h = (xn - x0) / k

    # Compute initial and final function values
    fx0 = f(x0)
    fxn = f(xn)

    # Initialize step with the sum of the first and last terms
    step = fx0 + fxn

    # Perform summation of intermediate terms
    for i in range(1, k):
        a = x0 + i * h
        step += 2 * f(a)

    # Compute the final result
    v = (h / 2) * step

    # Display the result
    print(f"Value of integration = {v:.6f}")

# Run the function
if __name__ == "__main__":
    trapezoidal_composite_rule()
"""

"""
# Define the function
def f(x):
    return 3 * (x ** 2) + 2 * x - 5

# Main function
def simpsons_one_third_rule():
    print("Sajan Bista Simpson's 1/3 Rule")

    # Input lower and upper limits
    x0 = float(input("Enter the lower limit: "))
    xn = float(input("Enter the upper limit: "))

    # Input number of segments
    k = int(input("Enter the number of segments (must be even): "))

    # Check if k is even
    if k % 2 != 0:
        print("Number of segments must be even for Simpson's 1/3 rule.")
        return

    # Calculate step size
    h = (xn - x0) / k

    # Compute initial and final function values
    fx0 = f(x0)
    fxn = f(xn)

    # Initialize term with the sum of the first and last terms
    term = fx0 + fxn

    # Summation for odd terms
    for i in range(1, k, 2):
        a = x0 + i * h
        term += 4 * f(a)

    # Summation for even terms
    for i in range(2, k, 2):
        a = x0 + i * h
        term += 2 * f(a)

    # Compute the final result
    v = (h / 3) * term

    # Display the result
    print(f"The output of Simpson's 1/3 rule = {v:.6f}")

# Run the function
if __name__ == "__main__":
    simpsons_one_third_rule()
"""
"""
# Define the function
def f(x):
    return 3 * (x ** 2) + 2 * x - 5

# Main function
def simpsons_three_eighth_rule():
    print("Sajan Bista Simpson's 3/8 Rule")

    # Input lower and upper limits
    x0 = float(input("Enter the lower limit: "))
    xn = float(input("Enter the upper limit: "))

    # Input number of segments
    k = int(input("Enter the number of segments (must be a multiple of 3): "))

    # Check if k is a multiple of 3
    if k % 3 != 0:
        print("Number of segments must be a multiple of 3 for Simpson's 3/8 rule.")
        return

    # Calculate step size
    h = (xn - x0) / k

    # Compute initial and final function values
    fx0 = f(x0)
    fxn = f(xn)

    # Initialize term with the sum of the first and last terms
    term = fx0 + fxn

    # Summation of intermediate terms
    for i in range(1, k):
        a = x0 + i * h
        if i % 3 != 0:
            term += 3 * f(a)
        else:
            term += 2 * f(a)

    # Compute the final result
    v = (3 / 8) * h * term

    # Display the result
    print(f"Value of integration = {v:.6f}")

# Run the function
if __name__ == "__main__":
    simpsons_three_eighth_rule()
"""
"""
# Define the function
def f(x):
    return x**3 + 1

# Main function
def gaussian_quadrature_two_point():
    print("Sajan Bista\nGaussian Quadrature Two-Point Rule")

    # Input lower and upper limits
    a = float(input("Enter the lower limit: "))
    b = float(input("Enter the upper limit: "))

    # Setting the values of the parameters
    c1 = c2 = 1  # Weights
    z1 = -0.57735
    z2 = 0.57735  # Roots of Legendre polynomial

    # Calculating xi
    x1 = (b - a) / 2 * z1 + (b + a) / 2
    x2 = (b - a) / 2 * z2 + (b + a) / 2

    # Calculating integral value
    v = (b - a) / 2 * ((f(x1) * c1) + (f(x2) * c2))

    # Displaying the result
    print(f"Value of integration = {v:.6f}")

# Run the function
if __name__ == "__main__":
    gaussian_quadrature_two_point()
"""
"""
# Define the function
def f(x):
    return x**3 + 1

# Main function
def romberg_integration():
    print("Sajan Bista\nRomberg Integration")

    # Input lower and upper limits
    x0 = float(input("Enter the lower limit: "))
    xn = float(input("Enter the upper limit: "))

    # Input required p and q for T(p, q)
    p = int(input("Enter p (rows): "))
    q = int(input("Enter q (columns): "))

    # Initialize T matrix
    T = [[0.0 for _ in range(q + 1)] for _ in range(p + 1)]

    # Step size
    h = xn - x0

    # T(0,0)
    T[0][0] = h / 2 * (f(x0) + f(xn))

    # Calculate T(i,0)
    for i in range(1, p + 1):
        sl = 2**(i - 1)
        sm = 0
        for k in range(1, int(sl) + 1):
            a = x0 + (2 * k - 1) * h / (2**i)
            sm += f(a)
        T[i][0] = T[i - 1][0] / 2 + sm * h / (2**i)

    # Calculate T(m+k, k)
    for c in range(1, p + 1):
        for k in range(1, min(c, q) + 1):
            m = c - k
            T[m + k][k] = (4**k * T[m + k][k - 1] - T[m + k - 1][k - 1]) / (4**k - 1)

    # Display the Romberg estimate
    print(f"Romberg Estimate of integration is = {T[p][q]:.6f}")

# Run the function
if __name__ == "__main__":
    romberg_integration()
"""
"""
import math

# Function to calculate factorial
def fact(n):
    if n == 1:
        return 1
    else:
        return n * fact(n - 1)

# Main function
def taylor_series():
    print("Sajan Bista\nTaylor Series\n")

    # Input initial values of x and y
    x0 = float(input("Enter the initial value of x: "))
    yx0 = float(input("Enter the initial value of y: "))

    # Input x at which the function is to be evaluated
    x = float(input("Enter the value of x at which the function is to be evaluated: "))

    # Calculating derivatives
    fdy = (x0)**2 + (yx0)**2  # First derivative
    sdy = 2 * x0 + 2 * yx0 * fdy  # Second derivative
    tdy = 2 + 2 * yx0 * sdy + 2 * fdy**2  # Third derivative

    # Calculating function value using Taylor series
    yx = (yx0 + (x - x0) * fdy 
          + ((x - x0)**2 * sdy) / fact(2) 
          + ((x - x0)**3 * tdy) / fact(3))

    # Displaying the result
    print(f"Function value at x = {x} is {yx:.6f}")

# Run the function
if __name__ == "__main__":
    taylor_series()
"""
"""
def f(x, y):
    return 2 * y / x

def euler_method():
    print("Sajan Bista\nEuler's Method\n")

    # Input initial values of x and y
    x0 = float(input("Enter the initial value of x: "))
    y0 = float(input("Enter the initial value of y: "))

    # Input the x value at which the function is to be evaluated
    xp = float(input("Enter the value of x at which the function is to be evaluated: "))

    # Input the step size
    h = float(input("Enter the step size: "))

    # Initialize x and y
    x = x0
    y = y0

    # Perform Euler's method
    while x < xp:
        y += f(x, y) * h
        x += h

    # Display the result
    print(f"Function value at x = {xp} is {y:.6f}")

# Run the function
if __name__ == "__main__":
    euler_method()
"""
"""
def f(x, y):
    return 2 * y / x

def heuns_method():
    print("Sajan Bista\nHeun's Method\n")

    # Input initial values of x and y
    x0 = float(input("Enter the initial value of x: "))
    y0 = float(input("Enter the initial value of y: "))

    # Input the x value at which the function is to be evaluated
    xp = float(input("Enter the value of x at which the function is to be evaluated: "))

    # Input the step size
    h = float(input("Enter the step size: "))

    # Initialize x and y
    x = x0
    y = y0

    # Perform Heun's method
    while x < xp:
        m1 = f(x, y)
        m2 = f(x + h, y + h * m1)
        y += (h / 2) * (m1 + m2)
        x += h

    # Display the result
    print(f"Function value at x = {xp} is {y:.6f}")

# Run the function
if __name__ == "__main__":
    heuns_method()
"""
"""
def f(x, y):
    return 2 * x + y

def runge_kutta():
    print("Sajan Bista\nFourth Order Runge-Kutta Method\n")

    # Input initial values of x and y
    x0 = float(input("Enter the initial value of x: "))
    y0 = float(input("Enter the initial value of y: "))

    # Input the x value at which the function is to be evaluated
    xp = float(input("Enter the value of x at which the function is to be evaluated: "))

    # Input the step size
    h = float(input("Enter the step size: "))

    # Initialize x and y
    x = x0
    y = y0

    # Perform Fourth Order Runge-Kutta Method
    while x < xp:
        m1 = f(x, y)
        m2 = f(x + 0.5 * h, y + 0.5 * h * m1)
        m3 = f(x + 0.5 * h, y + 0.5 * h * m2)
        m4 = f(x + h, y + h * m3)
        y += (h / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
        x += h

    # Display the result
    print(f"Function value at x = {xp} is {y:.6f}")

# Run the function
if __name__ == "__main__":
    runge_kutta()
"""
