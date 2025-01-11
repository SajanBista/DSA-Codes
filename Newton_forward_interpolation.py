import math

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def newtons_forward_interpolation():
    try:
        num_points = int(input("Enter the number of points: "))

        x_values = []
        y_values = []

        for i in range(num_points):
            x_val = float(input(f"Enter the value of x[{i}]: "))
            y_val = float(input(f"Enter the value of f(x[{i}]): "))
            x_values.append(x_val)
            y_values.append(y_val)

        interpolation_point = float(input("Enter the value of x for which you want to interpolate: "))

        interpolated_value = newtons_forward_interpolation(x_values, y_values, interpolation_point)
        print(f"Interpolated value at x = {interpolation_point} is {interpolated_value}")

    except ValueError:
        print("Invalid input. Please enter numerical values.")

newtons_forward_interpolation()