def divided_difference_table(x, y, n):
    """
    Calculates the divided difference table.

    Args:
        x: A list of x-values.
        y: A list of y-values.
        n: The number of data points.

    Returns:
        A 2D list representing the divided difference table.
    """

    table = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table

def newton_interpolation(x, y, value):
    """
    Performs Newton's Divided Difference Interpolation.

    Args:
        x: A list of x-values.
        y: A list of y-values.
        value: The value at which to interpolate.

    Returns:
        The interpolated value.
    """

    n = len(x)
    table = divided_difference_table(x, y, n)

    result = table[0][0]
    product = 1
    for i in range(1, n):
        product *= (value - x[i - 1])
        result += table[0][i] * product

    return result

def main():
    n = int(input("Enter the number of data points: "))

    x = []
    y = []

    for i in range(n):
        xi = float(input(f"Enter the value of x[{i}]: "))
        x.append(xi)
        yi = float(input(f"Enter the value of y[{i}]: "))
        y.append(yi)

    value = float(input("Enter the value of x for interpolation: "))

    interpolated_value = newton_interpolation(x, y, value)

    print(f"Interpolated value at x = {value} is {interpolated_value}")
if __name__ == "__main__":
    main()