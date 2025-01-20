"""
import math

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


def linear_regression():
    print("Sajan Bista\n")

    # Input the number of data points
    n = int(input("Enter the number of points n: "))

    # Input the x and y values
    x = []
    y = []
    print("Enter the values of x and y:")
    for i in range(n):
        xi, yi = map(float, input(f"x[{i}], y[{i}]: ").split())
        x.append(xi)
        y.append(yi)

    # Calculate summations
    sx = sum(x)
    sy = sum(y)
    sxy = sum(x[i] * y[i] for i in range(n))
    sx2 = sum(x[i] ** 2 for i in range(n))

    # Compute the slope (b) and intercept (a)
    b = ((n * sxy) - (sx * sy)) / ((n * sx2) - (sx ** 2))
    a = (sy / n) - (b * (sx / n))

    # Display the fitted line equation
    print(f"\nFitted line equation: y = {a:.2f} + {b:.2f} * x")


# Run the function
if __name__ == "__main__":
    linear_regression()


import math

def exponential_regression():
    print("Sajan Bista\n")

    # Input the number of data points
    n = int(input("Enter the number of points n: "))

    # Input the x and y values
    x = []
    y = []
    log_y = []
    print("Enter the values of x and y:")
    for i in range(n):
        xi, yi = map(float, input(f"x[{i}], y[{i}]: ").split())
        if yi <= 0:
            print("Error: y values must be positive for exponential regression.")
            return
        x.append(xi)
        y.append(yi)
        log_y.append(math.log(yi))  # Compute ln(y)

    # Calculate summations
    sx = sum(x)
    sy = sum(log_y)
    sxy = sum(x[i] * log_y[i] for i in range(n))
    sx2 = sum(x[i] ** 2 for i in range(n))

    # Compute b and A
    b = ((n * sxy) - (sx * sy)) / ((n * sx2) - (sx ** 2))
    A = (sy / n) - (b * (sx / n))

    # Compute a = exp(A)
    a = math.exp(A)

    # Display the fitted exponential equation
    print(f"\nFitted exponential equation: y = {a:.2f} * e^({b:.2f} * x)")


# Run the function
if __name__ == "__main__":
    exponential_regression()


import numpy as np

def main():
    print("Sajan Bista\n")

    # Input the number of data points
    n = int(input("Enter the number of data points (n): "))

    # Input the degree of the polynomial
    degree = int(input("Enter the degree of the polynomial: "))

    # Input x and y values
    x = []
    y = []
    print("Enter the values of x and y:")
    for i in range(n):
        xi, yi = map(float, input(f"x[{i}], y[{i}]: ").split())
        x.append(xi)
        y.append(yi)

    x = np.array(x)
    y = np.array(y)

    # Initialize summations for X and B
    X = np.zeros(2 * degree + 1)
    for i in range(2 * degree + 1):
        X[i] = np.sum(x ** i)

    B = np.zeros(degree + 1)
    for i in range(degree + 1):
        B[i] = np.sum((x ** i) * y)

    # Construct the augmented matrix
    A = np.zeros((degree + 1, degree + 2))
    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = X[i + j]
        A[i, -1] = B[i]

    # Perform Gaussian elimination
    for i in range(degree + 1):
        for j in range(degree + 1):
            if j != i:
                ratio = A[j, i] / A[i, i]
                A[j, :] -= ratio * A[i, :]

    # Extract coefficients
    coeff = np.zeros(degree + 1)
    for i in range(degree + 1):
        coeff[i] = A[i, -1] / A[i, i]

    # Display the polynomial equation
    print("\nThe fitted polynomial is:")
    print("y = ", end="")
    for i in range(degree + 1):
        if i == 0:
            print(f"{coeff[i]:.4f}", end="")
        else:
            print(f" + {coeff[i]:.4f}*x^{i}", end="")
    print()

if __name__ == "__main__":
    main()


import numpy as np

def gauss_elimination(a, n):
    # Forward Elimination
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = a[i, k] / a[k, k]
            a[i, k:] -= factor * a[k, k:]

    # Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (a[i, -1] - np.dot(a[i, i + 1:n], x[i + 1:])) / a[i, i]

    # Display the solution
    print("\nThe solution is:")
    for i in range(n):
        print(f"x[{i + 1}] = {x[i]:.4f}")

def main():
    print("Sajan Bista\n")

    # Input number of variables
    n = int(input("Enter the number of variables: "))

    # Input augmented matrix
    a = np.zeros((n, n + 1))
    print("Enter the augmented matrix (coefficients and constants):")
    for i in range(n):
        a[i] = list(map(float, input(f"Row {i + 1}: ").split()))

    # Perform Gauss Elimination
    gauss_elimination(a, n)

if __name__ == "__main__":
    main()
"""
"""
import numpy as np

def gauss_jordan(a, n):
    # Convert matrix to reduced row-echelon form
    for i in range(n):
        # Make the diagonal element 1
        a[i] /= a[i, i]
        # Make all other elements in the column 0
        for k in range(n):
            if k != i:
                a[k] -= a[k, i] * a[i]

    # Display the solution
    print("\nThe solution is:")
    for i in range(n):
        print(f"x[{i + 1}] = {a[i, -1]:.4f}")

def main():
    print("Sajan Bista\n")

    # Input number of variables
    n = int(input("Enter the number of variables: "))

    # Input augmented matrix
    a = np.zeros((n, n + 1))
    print("Enter the augmented matrix (coefficients and constants):")
    for i in range(n):
        a[i] = list(map(float, input(f"Row {i + 1}: ").split()))

    # Perform Gauss-Jordan Elimination
    gauss_jordan(a, n)

if __name__ == "__main__":
    main()

import numpy as np

def gauss_jordan_inverse(a):
    n = a.shape[0]
    # Augment matrix with identity matrix
    a = np.hstack((a, np.eye(n)))

    # Convert matrix to reduced row-echelon form
    for i in range(n):
        a[i] /= a[i, i]
        for k in range(n):
            if k != i:
                a[k] -= a[k, i] * a[i]

    # Extract inverse matrix
    return a[:, n:]

def main():
    print("Sajan Bista\n")

    # Input matrix size
    n = int(input("Enter the order of the matrix: "))

    # Input matrix
    print("Enter the elements of the matrix row by row:")
    a = np.array([list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)])

    # Compute inverse using Gauss-Jordan
    inverse = gauss_jordan_inverse(a)

    # Display the inverse matrix
    print("\nThe inverse of the matrix is:")
    for row in inverse:
        print(" ".join(f"{x:.4f}" for x in row))

if __name__ == "__main__":
    main()


import numpy as np

def jacobi_method(A, b, tolerance=1e-10, max_iterations=100):
    n = len(A)
    x = np.zeros(n)  # Initial guess (can be adjusted)
    x_new = np.zeros(n)

    for iteration in range(max_iterations):
        for i in range(n):
            # Calculate the sum excluding the diagonal element
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            print(f"Converged in {iteration + 1} iterations.")
            return x_new

        x = x_new.copy()

    print("Maximum iterations reached without convergence.")
    return x_new

def main():
    print("Sajan Bista")

    # Input matrix size
    n = int(input("Enter the order of the matrix: "))

    # Input coefficient matrix A
    print("Enter the elements of the coefficient matrix A row by row:")
    A = np.array([list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)])

    # Input constant vector b
    print("Enter the elements of the constant vector b:")
    b = np.array([float(input(f"b[{i + 1}]: ")) for i in range(n)])

    # Solve using Jacobi method
    solution = jacobi_method(A, b)

    # Output the solution
    print("\nSolution Vector:")
    print(" ".join(f"{xi:.6f}" for xi in solution))

if __name__ == "__main__":
    main()

import numpy as np

def gauss_seidel(A, b, tolerance=1e-6, max_iterations=100):
    n = len(A)
    x = np.zeros(n)  # Initial guess (can be adjusted)

    for iteration in range(max_iterations):
        x_new = np.copy(x)
        
        for i in range(n):
            # Calculate the sum excluding the diagonal element
            sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            print(f"Converged in {iteration + 1} iterations.")
            return x_new

        x = x_new

    print("Maximum iterations reached without convergence.")
    return x

def main():
    print("Sajan Bista")

    # Input matrix size
    n = int(input("Enter the number of variables (n): "))

    # Input coefficient matrix A
    print("Enter the elements of matrix A row by row:")
    A = np.array([list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)])

    # Input constant vector b
    print("Enter the elements of the constant vector b:")
    b = np.array([float(input(f"b[{i + 1}]: ")) for i in range(n)])

    # Input maximum iterations
    max_iterations = int(input("Enter the maximum number of iterations: "))

    # Solve using Gauss-Seidel method
    solution = gauss_seidel(A, b, max_iterations=max_iterations)

    # Output the solution
    print("\nSolution Vector:")
    print(" ".join(f"x[{i + 1}] = {xi:.6f}" for i, xi in enumerate(solution)))

if __name__ == "__main__":
    main()

import numpy as np

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Upper Triangular Matrix U
        for k in range(i, n):
            sum_u = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_u

        # Lower Triangular Matrix L
        L[i][i] = 1  # Diagonal elements of L are 1
        for k in range(i + 1, n):
            sum_l = sum(L[k][j] * U[j][i] for j in range(i))
            L[k][i] = (A[k][i] - sum_l) / U[i][i]

    return L, U

def forward_substitution(L, b):
    n = len(L)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    
    return y

def backward_substitution(U, y):
    n = len(U)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    
    return x

def solve_system_with_lu(A, b):
    # Step 1: LU Decomposition
    L, U = lu_decomposition(A)

    # Step 2: Solve L * y = b using forward substitution
    y = forward_substitution(L, b)

    # Step 3: Solve U * x = y using backward substitution
    x = backward_substitution(U, y)

    return x

def main():
    print("Sajan Bista")

    # Input matrix size
    n = int(input("Enter the number of variables (n): "))

    # Input coefficient matrix A
    print("Enter the elements of matrix A row by row:")
    A = np.array([list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)])

    # Input constant vector b
    print("Enter the elements of the constant vector b:")
    b = np.array([float(input(f"b[{i + 1}]: ")) for i in range(n)])

    # Solve the system using LU decomposition
    solution = solve_system_with_lu(A, b)

    # Output the solution
    print("\nSolution Vector:")
    print(" ".join(f"x[{i + 1}] = {xi:.6f}" for i, xi in enumerate(solution)))

if __name__ == "__main__":
    main()

import numpy as np

def gauss_jordan(A, b):
    n = len(A)
    # Augment matrix A with vector b
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

    # Perform Gauss-Jordan elimination
    for i in range(n):
        # Make the diagonal element 1 by dividing the row by A[i][i]
        diag_element = augmented_matrix[i][i]
        if diag_element == 0:
            raise ValueError("Matrix is singular or not invertible.")
        augmented_matrix[i] = augmented_matrix[i] / diag_element

        # Make all other elements in the column 0
        for j in range(n):
            if i != j:
                factor = augmented_matrix[j][i]
                augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]

    # Extract the solution vector from the augmented matrix
    x = augmented_matrix[:, -1]
    return x

def main():
    print("Sajan Bista")

    # Input matrix size
    n = int(input("Enter the number of variables (n): "))

    # Input coefficient matrix A
    print("Enter the elements of matrix A row by row:")
    A = np.array([list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)])

    # Input constant vector b
    print("Enter the elements of the constant vector b:")
    b = np.array([float(input(f"b[{i + 1}]: ")) for i in range(n)])

    # Solve using Gauss-Jordan method
    solution = gauss_jordan(A, b)

    # Output the solution
    print("\nSolution Vector:")
    print(" ".join(f"x[{i + 1}] = {xi:.6f}" for i, xi in enumerate(solution)))

if __name__ == "__main__":
    main()
"""
def trapezoidal_rule(f, a, b, n):
    """
    Perform numerical integration using the Trapezoidal Rule.

    Parameters:
    f : function
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n : int
        The number of subintervals.

    Returns:
    float
        The approximate integral of the function over [a, b].
    
    h = (b - a) / n  # Width of each subinterval
    integral = 0.5 * (f(a) + f(b))  # First and last terms

    for i in range(1, n):
        x = a + i * h
        integral += f(x)

    integral *= h
    return integral

def main():
    print("Sajan Bista")

    # Define the function to integrate
    f = lambda x: x**2  # Example: f(x) = x^2

    # Input integration limits and number of intervals
    a = float(input("Enter the lower limit of integration (a): "))
    b = float(input("Enter the upper limit of integration (b): "))
    n = int(input("Enter the number of subintervals (n): "))

    # Calculate the integral
    result = trapezoidal_rule(f, a, b, n)

    # Output the result
    print(f"\nThe approximate integral of the function over [{a}, {b}] is: {result:.6f}")

if __name__ == "__main__":
    main()

def simpson_one_third_rule(f, a, b, n):
    
    Perform numerical integration using Simpson's 1/3 Rule.

    Parameters:
    f : function
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n : int
        The number of subintervals (must be even).

    Returns:
    float
        The approximate integral of the function over [a, b].
    
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even for Simpson's 1/3 Rule.")

    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    fx = [f(xi) for xi in x]

    integral = fx[0] + fx[-1]  # First and last terms
    for i in range(1, n):
        if i % 2 == 0:
            integral += 2 * fx[i]
        else:
            integral += 4 * fx[i]

    integral *= h / 3
    return integral

def main():
    print("Sajan Bista")

    # Define the function to integrate
    f = lambda x: x**2  # Example: f(x) = x^2

    # Input integration limits and number of intervals
    a = float(input("Enter the lower limit of integration (a): "))
    b = float(input("Enter the upper limit of integration (b): "))
    n = int(input("Enter the number of subintervals (n, must be even): "))

    # Calculate the integral using Simpson's 1/3 Rule
    try:
        result = simpson_one_third_rule(f, a, b, n)
        print(f"\nThe approximate integral of the function over [{a}, {b}] is: {result:.6f}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
"""

def simpson_three_eighths_rule(f, a, b, n):
    """
    Perform numerical integration using Simpson's 3/8 Rule.

    Parameters:
    f : function
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n : int
        The number of subintervals (must be a multiple of 3).

    Returns:
    float
        The approximate integral of the function over [a, b].
    
    if n % 3 != 0:
        raise ValueError("Number of subintervals (n) must be a multiple of 3 for Simpson's 3/8 Rule.")

    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    fx = [f(xi) for xi in x]

    integral = fx[0] + fx[-1]  # First and last terms
    for i in range(1, n):
        if i % 3 == 0:
            integral += 2 * fx[i]
        else:
            integral += 3 * fx[i]

    integral *= (3 * h) / 8
    return integral

def main():
    print("Sajan Bista")

    # Define the function to integrate
    f = lambda x: x**2  # Example: f(x) = x^2

    # Input integration limits and number of intervals
    a = float(input("Enter the lower limit of integration (a): "))
    b = float(input("Enter the upper limit of integration (b): "))
    n = int(input("Enter the number of subintervals (n, must be a multiple of 3): "))

    # Calculate the integral using Simpson's 3/8 Rule
    try:
        result = simpson_three_eighths_rule(f, a, b, n)
        print(f"\nThe approximate integral of the function over [{a}, {b}] is: {result:.6f}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

def euler_method(f, x0, y0, z0, h, xn):
    
    Euler's method for solving an ODE system y'' = f(x, y, y').
    Returns the final values of y and y' at xn.

    Parameters:
    f  : Function representing y'' = f(x, y, y')
    x0 : Initial x
    y0 : Initial y
    z0 : Initial y' (dy/dx)
    h  : Step size
    xn : Final x value
    
    x, y, z = x0, y0, z0
    while x < xn:
        y_next = y + h * z
        z_next = z + h * f(x, y, z)
        x += h
        y, z = y_next, z_next
    return y, z

def shooting_method(f, x0, xn, y0, yn, z_guess1, z_guess2, h, tolerance=1e-6):
    
    Solves a boundary value problem using the shooting method.

    Parameters:
    f         : Function representing y'' = f(x, y, y')
    x0, xn    : Boundary x values
    y0, yn    : Boundary y values
    z_guess1  : First guess for y'(x0)
    z_guess2  : Second guess for y'(x0)
    h         : Step size
    tolerance : Convergence tolerance

    Returns:
    Final solution for z(x0).
    
    # Solve with the first guess
    y1, _ = euler_method(f, x0, y0, z_guess1, h, xn)

    # Solve with the second guess
    y2, _ = euler_method(f, x0, y0, z_guess2, h, xn)

    # Iterate until the solution converges
    while abs(y1 - yn) > tolerance:
        # Linear interpolation to adjust the guess
        z_new = z_guess2 - (y2 - yn) * (z_guess2 - z_guess1) / (y2 - y1)

        # Update guesses
        z_guess1, z_guess2 = z_guess2, z_new

        # Solve with the new guess
        y1, _ = euler_method(f, x0, y0, z_guess1, h, xn)
        y2, _ = euler_method(f, x0, y0, z_guess2, h, xn)

    return z_guess2

def main():
    print("Sajan Bista - Shooting Method Implementation")

    # Define the ODE y'' = f(x, y, y')
    def f(x, y, z):
        return -2 * z + y + x  # Example: y'' + 2y' - y = x

    # Input boundary conditions and step size
    x0 = float(input("Enter the initial x value (x0): "))
    xn = float(input("Enter the final x value (xn): "))
    y0 = float(input("Enter the boundary condition y(x0): "))
    yn = float(input("Enter the boundary condition y(xn): "))
    h = float(input("Enter the step size (h): "))

    # Input initial guesses for y'(x0)
    z_guess1 = float(input("Enter the first guess for y'(x0): "))
    z_guess2 = float(input("Enter the second guess for y'(x0): "))

    # Solve the boundary value problem
    z_solution = shooting_method(f, x0, xn, y0, yn, z_guess1, z_guess2, h)
    print(f"The estimated value of y'(x0) is: {z_solution:.6f}")

if __name__ == "__main__":
    main()
"""
import numpy as np
import scipy.integrate as integrate

# Laplace Transform of Exponential Function e^(-at)
def laplace_exp(a, s):
    """
    Laplace Transform of e^(-at).
    Formula: L{e^(-at)} = 1 / (s + a)
    """
    return 1 / (s + a)

# Laplace Transform of t^n (Polynomial Function)
def laplace_polynomial(n, s):
    """
    Laplace Transform of t^n.
    Formula: L{t^n} = n! / s^(n+1)
    """
    return np.math.factorial(n) / s**(n+1)

# Numerical Laplace Transform using Integration for general functions
def laplace_integral(func, s):
    """
    Numerical Laplace transform of a function using integration.
    """
    result, error = integrate.quad(lambda t: np.exp(-s * t) * func(t), 0, np.inf)
    return result

# Example function: e^(-2t)
def example_func(t):
    return np.exp(-2 * t)

# Calculate the Laplace transform of e^(-2t) at s = 3
s = 3
result_exp = laplace_exp(2, s)
print(f"Laplace transform of e^(-2t) at s = {s}: {result_exp}")

# Calculate the Laplace transform of t^2 (n=2) at s = 3
result_polynomial = laplace_polynomial(2, s)
print(f"Laplace transform of t^2 at s = {s}: {result_polynomial}")

# Numerical calculation of the Laplace transform of e^(-2t) using integration
result_integral = laplace_integral(example_func, s)
print(f"Numerical Laplace transform of e^(-2t) at s = {s}: {result_integral}")
