"""import math
def larange_interpolation(x_point,y_point,x):
    n= len(x_point)
    result = 0.0

    for i in range(n):
        term = y_point[i]
        for j in range(n):
            if j!=i:
                term = term * (x-x_point[j])/(x-x_point[i]-x_point[j])
        result += term

    return result
n = int(input("Enter the number of data points:"))

x_points=[]
y_points=[]
for i in range(n):
    x_point=float(input(f"enter x[{i}]:"))
    y_point=float(input(f"enter y[{i}]:"))
    x_points.append(x_point)
    y_points.append(y_point)

x_point=float(input("enter the value of x for interpolation:"))
interpolated_value=larange_interpolation(x_point,y_point,x_value)
print(f"the interpolation value at x ={x_value} is is {interpolated_value}")"""

def lagrange_interpolation(x_points, f_points, x_interp):
    """
    Performs Lagrange interpolation on a set of data points.

    Args:
        x_points: A list of x-coordinates.
        y_points: A list of corresponding y-coordinates.
        x_interp: The x-value at which to interpolate.

    Returns:
        The interpolated y-value.
    """

    n = len(x_points)
    result = 0.0

    for i in range(n):
        term = f_points[i]
        for j in range(n):
            if j != i:
                term *= (x_interp - x_points[j]) / (x_points[i] - x_points[j])
        result += term

    return result

def main():
    # Get user input
    n = int(input("Enter the number of data points: "))

    x_points = []
    f_points = []
    for i in range(n):
        x_point = float(input(f"Enter x[{i}]: "))
        f_point = float(input(f"Enter y[{i}]: "))
        x_points.append(x_point)
        f_points.append(f_point)

    x_interp = float(input("Enter the value of x for interpolation: "))

    # Perform Lagrange interpolation
    interpolated_value = lagrange_interpolation(x_points, f_points, x_interp)
    print(f"The interpolated value at x = {x_interp} is {interpolated_value}")

if __name__ == "__main__":#dunder method.
    main()
