/*#include <stdio.h>

int main() {
    printf("sharad bista\n");
    int n;
    
    // Input the number of data points
    printf("Enter number of data points: ");
    scanf("%d", &n);

    double x[n], y[n];
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    
    // Input data points
    printf("Enter x and y values:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d]: ", i);
        scanf("%lf", &x[i]);
        printf("y[%d]: ", i);
        scanf("%lf", &y[i]);

        // Summation calculations
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    // Compute slope (m) and intercept (c) without pointers
    double m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double c = (sum_y - m * sum_x) / n;

    // Output the result
    printf("\nLinear Regression Equation: y = %.4fx + %.4f\n", m, c);

    return 0;
}


#include <stdio.h>
#include <math.h>

int main() {
    printf("sharad bista\n");
    int n;

    // Input the number of data points
    printf("Enter number of data points: ");
    scanf("%d", &n);

    double x[n], y[n], log_y[n];
    double sum_x = 0, sum_log_y = 0, sum_x_log_y = 0, sum_x2 = 0;

    // Input x and y values
    printf("Enter x and y values:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d]: ", i);
        scanf("%lf", &x[i]);
        printf("y[%d]: ", i);
        scanf("%lf", &y[i]);

        // Convert y to ln(y)
        log_y[i] = log(y[i]);

        // Compute summations
        sum_x += x[i];
        sum_log_y += log_y[i];
        sum_x_log_y += x[i] * log_y[i];
        sum_x2 += x[i] * x[i];
    }

    // Compute coefficients A and b using linear regression formulas
    double b = (n * sum_x_log_y - sum_x * sum_log_y) / (n * sum_x2 - sum_x * sum_x);
    double A = (sum_log_y - b * sum_x) / n;

    // Convert A back to a
    double a = exp(A);

    // Output the exponential regression equation
    printf("\nExponential Regression Equation: y = %.4f * e^(%.4f * x)\n", a, b);

    return 0;
}

#include <stdio.h>
#include <math.h>

#define MAX_ORDER 5  // Maximum polynomial order

int main() {
    printf(" Sharad Bista\n");
    int n, m;

    // Input the number of data points
    printf("Enter number of data points: ");
    scanf("%d", &n);

    double x[n], y[n];

    // Input data points
    printf("Enter x and y values:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d]: ", i);
        scanf("%lf", &x[i]);
        printf("y[%d]: ", i);
        scanf("%lf", &y[i]);
    }

    // Input polynomial degree
    printf("Enter polynomial degree (max %d): ", MAX_ORDER);
    scanf("%d", &m);

    if (m >= n) {
        printf("Degree must be less than number of data points!\n");
        return 1;
    }

    // Construct Normal Equation Matrix
    double X[10][10] = {0}, B[10] = {0}, A[10];

    // Compute sum values for normal equations
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= m; j++) {
            X[i][j] = 0;
            for (int k = 0; k < n; k++) {
                X[i][j] += pow(x[k], i + j);
            }
        }

        B[i] = 0;
        for (int k = 0; k < n; k++) {
            B[i] += y[k] * pow(x[k], i);
        }
    }

    // Solve using Gaussian Elimination
    for (int i = 0; i <= m; i++) {
        for (int k = i + 1; k <= m; k++) {
            double factor = X[k][i] / X[i][i];
            for (int j = 0; j <= m; j++) {
                X[k][j] -= factor * X[i][j];
            }
            B[k] -= factor * B[i];
        }
    }

    for (int i = m; i >= 0; i--) {
        A[i] = B[i];
        for (int j = i + 1; j <= m; j++) {
            A[i] -= X[i][j] * A[j];
        }
        A[i] /= X[i][i];
    }

    // Print polynomial equation
    printf("\nPolynomial Regression Equation:\n");
    printf("y = ");
    for (int i = 0; i <= m; i++) {
        printf("%.4f", A[i]);
        if (i > 0) {
            printf(" * x^%d", i);
        }
        if (i < m) {
            printf(" + ");
        }
    }
    printf("\n");

    return 0;
}


#include <stdio.h>

#define MAX 10

// Function to perform the forward elimination
void forwardElimination(float matrix[MAX][MAX], float results[MAX], int n) {
    for (int i = 0; i < n; i++) {
        // Make the diagonal element 1 and eliminate below it
        for (int j = i + 1; j < n; j++) {
            float ratio = matrix[j][i] / matrix[i][i];
            for (int k = i; k < n; k++) {
                matrix[j][k] -= ratio * matrix[i][k];
            }
            results[j] -= ratio * results[i];
        }
    }
}

// Function to perform back substitution
void backSubstitution(float matrix[MAX][MAX], float results[MAX], int n) {
    for (int i = n - 1; i >= 0; i--) {
        // Start with the results[i] value
        for (int j = i + 1; j < n; j++) {
            results[i] -= matrix[i][j] * results[j];
        }
        // Divide by the diagonal element to get the solution
        results[i] /= matrix[i][i];
    }
}

// Function to display the matrix and results
void display(float matrix[MAX][MAX], float results[MAX], int n) {
    printf("\nThe solutions are: \n");
    for (int i = 0; i < n; i++) {
        printf("X%d = %.2f\n", i + 1, results[i]);
    }
}

int main() {
    printf(" Sharad Bista\n");
    int n;
    float matrix[MAX][MAX], results[MAX];

    // Input number of variables
    printf("Enter the number of variables: ");
    scanf("%d", &n);

    // Input augmented matrix
    printf("Enter the augmented matrix (coefficients and constant terms):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &matrix[i][j]);
        }
        scanf("%f", &results[i]); // The right-hand side values
    }

    // Perform Gauss Elimination
    forwardElimination(matrix, results, n);
    backSubstitution(matrix, results, n);

    // Display the result
    display(matrix, results, n);

    return 0;
}


#include <stdio.h>

#define MAX 10

// Function to perform Gauss-Jordan Elimination
void gaussJordan(float matrix[MAX][MAX], float results[MAX], int n) {
    for (int i = 0; i < n; i++) {
        // Make the diagonal element 1 and eliminate the other elements in the column
        float pivot = matrix[i][i];
        for (int j = 0; j < n; j++) {
            matrix[i][j] /= pivot;
        }
        results[i] /= pivot;

        for (int j = 0; j < n; j++) {
            if (j != i) {
                float ratio = matrix[j][i];
                for (int k = 0; k < n; k++) {
                    matrix[j][k] -= ratio * matrix[i][k];
                }
                results[j] -= ratio * results[i];
            }
        }
    }
}

// Function to display the results
void displayResults(float results[MAX], int n) {
    printf("\nThe solutions are: \n");
    for (int i = 0; i < n; i++) {
        printf("X%d = %.2f\n", i + 1, results[i]);
    }
}

int main() {
    int n;
    float matrix[MAX][MAX], results[MAX];

    // Input number of variables
    printf("Enter the number of variables: ");
    scanf("%d", &n);

    // Input augmented matrix
    printf("Enter the augmented matrix (coefficients and constant terms):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &matrix[i][j]);
        }
        scanf("%f", &results[i]); // The right-hand side values
    }

    // Perform Gauss-Jordan Elimination
    gaussJordan(matrix, results, n);

    // Display the result
    displayResults(results, n);

    return 0;
}

#include <stdio.h>

#define MAX 10

// Function to perform Gauss-Jordan Elimination for matrix inversion
void gaussJordanInvert(float matrix[MAX][MAX], int n) {
    float augmented[MAX][MAX * 2];

    // Create an augmented matrix [A | I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i][j] = matrix[i][j];
            augmented[i][j + n] = (i == j) ? 1.0 : 0.0;  // Identity matrix
        }
    }

    // Apply Gauss-Jordan Elimination
    for (int i = 0; i < n; i++) {
        // Make the diagonal element 1
        float pivot = augmented[i][i];
        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }

        // Make the elements in the current column 0 except the diagonal
        for (int j = 0; j < n; j++) {
            if (j != i) {
                float ratio = augmented[j][i];
                for (int k = 0; k < 2 * n; k++) {
                    augmented[j][k] -= ratio * augmented[i][k];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = n; j < 2 * n; j++) {
            matrix[i][j - n] = augmented[i][j];  // Store the inverse in the original matrix
        }
    }
}

// Function to display the matrix
void displayMatrix(float matrix[MAX][MAX], int n) {
    printf("\nThe inverted matrix is: \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    printf("Sharad Bista\n");
    int n;
    float matrix[MAX][MAX];

    // Input the size of the matrix
    printf("Enter the size of the matrix: ");
    scanf("%d", &n);

    // Input the matrix
    printf("Enter the matrix (coefficients):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &matrix[i][j]);
        }
    }

    // Perform matrix inversion using Gauss-Jordan method
    gaussJordanInvert(matrix, n);

    // Display the inverted matrix
    displayMatrix(matrix, n);

    return 0;
}


#include <stdio.h>

#define MAX 10

// Function to perform LU Decomposition
int luDecompose(float matrix[MAX][MAX], float lower[MAX][MAX], float upper[MAX][MAX], int n) {
    for (int i = 0; i < n; i++) {
        // Upper Triangular matrix
        for (int j = i; j < n; j++) {
            upper[i][j] = matrix[i][j];
            for (int k = 0; k < i; k++) {
                upper[i][j] -= lower[i][k] * upper[k][j];
            }
        }

        // Lower Triangular matrix
        for (int j = i; j < n; j++) {
            if (i == j)
                lower[i][i] = 1;  // Diagonal is set to 1
            else {
                lower[j][i] = matrix[j][i];
                for (int k = 0; k < i; k++) {
                    lower[j][i] -= lower[j][k] * upper[k][i];
                }
                lower[j][i] /= upper[i][i];
            }
        }
    }
    return 1;
}

// Function to display a matrix
void displayMatrix(float matrix[MAX][MAX], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    printf("sharad bista\n");
    int n;
    float matrix[MAX][MAX], lower[MAX][MAX] = {0}, upper[MAX][MAX] = {0};

    // Input the size of the matrix
    printf("Enter the size of the matrix: ");
    scanf("%d", &n);

    // Input the matrix
    printf("Enter the matrix (coefficients):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &matrix[i][j]);
        }
    }

    // Perform LU Decomposition
    if (luDecompose(matrix, lower, upper, n)) {
        // Display Lower Triangular Matrix
        printf("\nLower Triangular Matrix:\n");
        displayMatrix(lower, n);

        // Display Upper Triangular Matrix
        printf("\nUpper Triangular Matrix:\n");
        displayMatrix(upper, n);
    } else {
        printf("\nMatrix cannot be decomposed.\n");
    }

    return 0;
}
 
#include <stdio.h>
#include <math.h>

#define MAX 10
#define TOLERANCE 0.0001
#define MAX_ITER 1000

// Function to perform Jacobi Iteration
void jacobiIteration(float matrix[MAX][MAX], float b[MAX], float x[MAX], int n) {
    float x_new[MAX];
    int iteration = 0;

    while (iteration < MAX_ITER) {
        // Perform one iteration of Jacobi method
        for (int i = 0; i < n; i++) {
            x_new[i] = b[i];
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    x_new[i] -= matrix[i][j] * x[j];
                }
            }
            x_new[i] /= matrix[i][i];
        }

        // Check for convergence
        float max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            max_diff = fmax(max_diff, fabs(x_new[i] - x[i]));
        }

        // Update solution vector and check if convergence criterion is met
        if (max_diff < TOLERANCE) {
            break;
        }

        // Update the current solution
        for (int i = 0; i < n; i++) {
            x[i] = x_new[i];
        }

        iteration++;
    }

    if (iteration == MAX_ITER) {
        printf("Maximum iterations reached.\n");
    }
}

// Function to display the result
void displayResult(float x[MAX], int n) {
    printf("\nThe solution is: \n");
    for (int i = 0; i < n; i++) {
        printf("X%d = %.4f\n", i + 1, x[i]);
    }
}

int main() {
    printf("Sharad Bista\n");
    int n;
    float matrix[MAX][MAX], b[MAX], x[MAX] = {0};

    // Input the size of the matrix
    printf("Enter the number of variables: ");
    scanf("%d", &n);

    // Input the coefficient matrix
    printf("Enter the coefficient matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &matrix[i][j]);
        }
    }

    // Input the right-hand side vector
    printf("Enter the right-hand side vector:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &b[i]);
    }

    // Perform Jacobi Iteration to solve the system
    jacobiIteration(matrix, b, x, n);

    // Display the result
    displayResult(x, n);

    return 0;
}

#include <stdio.h>
#include <math.h>

#define MAX 10
#define TOLERANCE 0.0001
#define MAX_ITER 1000

// Function to perform Gauss-Seidel Iteration
void gaussSeidel(float matrix[MAX][MAX], float b[MAX], float x[MAX], int n) {
    float x_new[MAX];
    int iteration = 0;

    // Initialize x_new with initial guess
    for (int i = 0; i < n; i++) {
        x_new[i] = 0.0;  // Initial guess for the solution
    }

    while (iteration < MAX_ITER) {
        // Perform one iteration of Gauss-Seidel method
        for (int i = 0; i < n; i++) {
            x_new[i] = b[i];
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    x_new[i] -= matrix[i][j] * x_new[j];  // Use updated values
                }
            }
            x_new[i] /= matrix[i][i];
        }

        // Check for convergence
        float max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            max_diff = fmax(max_diff, fabs(x_new[i] - x[i]));
        }

        // Update solution vector and check if convergence criterion is met
        if (max_diff < TOLERANCE) {
            break;
        }

        // Update the current solution
        for (int i = 0; i < n; i++) {
            x[i] = x_new[i];
        }

        iteration++;
    }

    if (iteration == MAX_ITER) {
        printf("Maximum iterations reached.\n");
    }
}

// Function to display the result
void displayResult(float x[MAX], int n) {
    printf("\nThe solution is: \n");
    for (int i = 0; i < n; i++) {
        printf("X%d = %.4f\n", i + 1, x[i]);
    }
}

int main() {
    printf("Sharad Bista\n");
    int n;
    float matrix[MAX][MAX], b[MAX], x[MAX] = {0};

    // Input the size of the matrix
    printf("Enter the number of variables: ");
    scanf("%d", &n);

    // Input the coefficient matrix
    printf("Enter the coefficient matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &matrix[i][j]);
        }
    }

    // Input the right-hand side vector
    printf("Enter the right-hand side vector:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &b[i]);
    }

    // Perform Gauss-Seidel Iteration to solve the system
    gaussSeidel(matrix, b, x, n);

    // Display the result
    displayResult(x, n);

    return 0;
}

#include <stdio.h>
#include <math.h>

#define MAX 10
#define TOLERANCE 0.0001
#define MAX_ITER 1000

// Forward Two-Point Method (Solving linear system of equations)
void forwardTwoPoint(float matrix[MAX][MAX], float b[MAX], float x[MAX], int n) {
    for (int i = 0; i < n; i++) {
        x[i] = b[i];
    }

    // Forward elimination
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            float factor = matrix[j][i] / matrix[i][i];
            for (int k = 0; k < n; k++) {
                matrix[j][k] -= factor * matrix[i][k];
            }
            b[j] -= factor * b[i];
        }
    }

    // Back substitution (calculate final values of x)
    for (int i = n - 1; i >= 0; i--) {
        float sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += matrix[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / matrix[i][i];
    }
}

// Backward Two-Point Method (Solving linear system of equations)
void backwardTwoPoint(float matrix[MAX][MAX], float b[MAX], float x[MAX], int n) {
    // Backward elimination
    for (int i = n - 1; i >= 1; i--) {
        for (int j = i - 1; j >= 0; j--) {
            float factor = matrix[j][i] / matrix[i][i];
            for (int k = 0; k < n; k++) {
                matrix[j][k] -= factor * matrix[i][k];
            }
            b[j] -= factor * b[i];
        }
    }

    // Forward substitution (calculate final values of x)
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < i; j++) {
            sum += matrix[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / matrix[i][i];
    }
}

// Function to display the result
void displayResult(float x[MAX], int n) {
    printf("\nThe solution is: \n");
    for (int i = 0; i < n; i++) {
        printf("X%d = %.4f\n", i + 1, x[i]);
    }
}

int main() {
    printf("Sharad Bista\n");
    int n;
    float matrix[MAX][MAX], b[MAX], x[MAX] = {0};

    // Input the size of the matrix
    printf("Enter the number of variables: ");
    scanf("%d", &n);

    // Input the coefficient matrix
    printf("Enter the coefficient matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &matrix[i][j]);
        }
    }

    // Input the right-hand side vector
    printf("Enter the right-hand side vector:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &b[i]);
    }

    // Choose which method to apply: Forward or Backward
    int choice;
    printf("Enter 1 for Forward Two-Point Method or 2 for Backward Two-Point Method: ");
    scanf("%d", &choice);

    if (choice == 1) {
        forwardTwoPoint(matrix, b, x, n);
    } else if (choice == 2) {
        backwardTwoPoint(matrix, b, x, n);
    } else {
        printf("Invalid choice.\n");
        return 0;
    }

    // Display the result
    displayResult(x, n);

    return 0;
}
 
#include <stdio.h>
#include <math.h>

#define MAX 10
#define TOLERANCE 0.0001
#define MAX_ITER 1000

// Three-Point Forward Method (Solving linear system of equations)
void threePointForward(float matrix[MAX][MAX], float b[MAX], float x[MAX], int n) {
    // Forward elimination (modified three-point approach)
    for (int i = 0; i < n - 2; i++) {
        for (int j = i + 1; j < n; j++) {
            float factor = matrix[j][i] / matrix[i][i];
            for (int k = 0; k < n; k++) {
                matrix[j][k] -= factor * matrix[i][k];
            }
            b[j] -= factor * b[i];
        }
    }

    // Back substitution (final computation for x values)
    for (int i = n - 1; i >= 0; i--) {
        float sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += matrix[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / matrix[i][i];
    }
}

// Three-Point Backward Method (Solving linear system of equations)
void threePointBackward(float matrix[MAX][MAX], float b[MAX], float x[MAX], int n) {
    // Backward elimination (modified three-point approach)
    for (int i = n - 1; i >= 2; i--) {
        for (int j = i - 1; j >= 0; j--) {
            float factor = matrix[j][i] / matrix[i][i];
            for (int k = 0; k < n; k++) {
                matrix[j][k] -= factor * matrix[i][k];
            }
            b[j] -= factor * b[i];
        }
    }

    // Forward substitution (final computation for x values)
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < i; j++) {
            sum += matrix[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / matrix[i][i];
    }
}

// Function to display the result
void displayResult(float x[MAX], int n) {
    printf("\nThe solution is: \n");
    for (int i = 0; i < n; i++) {
        printf("X%d = %.4f\n", i + 1, x[i]);
    }
}

int main() {
    printf("sharad bista\n");
    int n;
    float matrix[MAX][MAX], b[MAX], x[MAX] = {0};

    // Input the size of the matrix
    printf("Enter the number of variables: ");
    scanf("%d", &n);

    // Input the coefficient matrix
    printf("Enter the coefficient matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &matrix[i][j]);
        }
    }

    // Input the right-hand side vector
    printf("Enter the right-hand side vector:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &b[i]);
    }

    // Choose which method to apply: Forward or Backward
    int choice;
    printf("Enter 1 for Three-Point Forward Method or 2 for Three-Point Backward Method: ");
    scanf("%d", &choice);

    if (choice == 1) {
        threePointForward(matrix, b, x, n);
    } else if (choice == 2) {
        threePointBackward(matrix, b, x, n);
    } else {
        printf("Invalid choice.\n");
        return 0;
    }

    // Display the result
    displayResult(x, n);

    return 0;
}


#include <stdio.h>
#include <math.h>

#define MAX 10

// Function to calculate the forward difference
void forwardDifference(float x[], float y[], int n, float h, float point) {
    // Finding the index of the point closest to the given point
    int index = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] > point) {
            index = i - 1;
            break;
        }
    }

    // Calculating forward differences
    float forwardDiff[MAX][MAX];

    // Initializing the forward differences table
    for (int i = 0; i < n; i++) {
        forwardDiff[i][0] = y[i];
    }

    // Construct the forward difference table
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < n - j; i++) {
            forwardDiff[i][j] = forwardDiff[i+1][j-1] - forwardDiff[i][j-1];
        }
    }

    // Using Newton's Forward formula to estimate the derivative at the point
    float result = 0;
    float multiplier = 1;
    for (int i = 0; i < index; i++) {
        multiplier *= (point - x[i]);
    }

    for (int j = 0; j < index; j++) {
        result += forwardDiff[0][j] * multiplier / pow(h, j+1);
        multiplier *= (point - x[j]);
    }

    printf("The derivative using Forward Difference at %.2f is: %.4f\n", point, result);
}

// Function to calculate the backward difference
void backwardDifference(float x[], float y[], int n, float h, float point) {
    // Finding the index of the point closest to the given point
    int index = 0;
    for (int i = n-1; i >= 0; i--) {
        if (x[i] < point) {
            index = i + 1;
            break;
        }
    }

    // Calculating backward differences
    float backwardDiff[MAX][MAX];

    // Initializing the backward differences table
    for (int i = 0; i < n; i++) {
        backwardDiff[i][0] = y[i];
    }

    // Construct the backward difference table
    for (int j = 1; j < n; j++) {
        for (int i = n-1; i >= j; i--) {
            backwardDiff[i][j] = backwardDiff[i][j-1] - backwardDiff[i-1][j-1];
        }
    }

    // Using Newton's Backward formula to estimate the derivative at the point
    float result = 0;
    float multiplier = 1;
    for (int i = 0; i < index; i++) {
        multiplier *= (point - x[i]);
    }

    for (int j = 0; j < index; j++) {
        result += backwardDiff[n-1][j] * multiplier / pow(h, j+1);
        multiplier *= (point - x[j]);
    }

    printf("The derivative using Backward Difference at %.2f is: %.4f\n", point, result);
}

int main() {
    printf("Sharad Bista\n");
    int n;
    float x[MAX], y[MAX], point, h;

    // Input number of data points
    printf("Enter the number of data points: ");
    scanf("%d", &n);

    // Input the x values
    printf("Enter the x values:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &x[i]);
    }

    // Input the y values (function values at x)
    printf("Enter the y values:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &y[i]);
    }

    // Input step size
    printf("Enter the step size h: ");
    scanf("%f", &h);

    // Input the point at which to calculate the derivative
    printf("Enter the point at which to calculate the derivative: ");
    scanf("%f", &point);

    // Call both methods
    forwardDifference(x, y, n, h, point);
    backwardDifference(x, y, n, h, point);

    return 0;
}

//composite trapezoidal rule
#include<stdio.h>
#include<math.h>
float f(float x){
    return 3*(x)*(x)+2*(x)-5;
    
}
int main(void){
    printf(" Sharad Bista\n ");
    float x0,xn,fxn,fx0,h,step,a,v;
    int i,k;
    printf("enter the lower and upper bound\t");
    scanf("%f%f",&x0,&xn);
    printf("enter the number of segment");
    scanf("%d",&k);
    
    h=(xn-x0)/k;
    
    
    fx0=f(x0);
    fxn=f(xn);
    
    
    step = f(x0)+ f(xn);
    
    for(i=1;i<k;i++){
        a=x0+i*h;
        step= step +2*f(a);
        
    }
    v= h/2* step;
    
    
    
    printf("Value of integration = %f\n",v);
    return 0;
}


//simpson's 1/3 rule

//simpson's composite rule for 1/3


#include<stdio.h>
#include<math.h>

float f(float x){
    return 3*(x)*(x)+2*(x)-5;
    
}

int main(void){
    
    printf("Sharad Bista\n");
    printf("enter the lower and upper limit \n");
    float xn,x0,fx0,fxn,h,term,a,v;
    int k,i;
    scanf("%f%f",&x0,&xn);
    
    printf("enter the number of segements\n");
    scanf("%d",&k);
    
    h=(xn-x0)/k;
    
    fx0=f(x0);
    fxn=f(xn);
    
    term = f(x0)+f(xn);
    for(i=1;i<=k-1;i+=2){
        a= x0+i*h;
        term = term+4*f(a);
        
    }
    for(i=1;i<=k-2;i+=2){
        a= x0+i*h;
        term = term+2*f(a);
        
    }
    v = h/3 * term;
    
    printf("the output of the simpson's 1/3 rule %f ",v);
    
}
 

#include<stdio.h>
#include<math.h>
float f(float x){
    return 3*(x)*(x)+2*(x)-5;
    
}
int main(void){
    printf("Sharad Bista\n");
    float xn, x0,fxn,fx0,h,term,a,v;
    int k,i;
    printf("enter the lower and upper limit\n");
    scanf("%f%f",&x0,&xn);
    
    printf("enter the number of segment \n");
    scanf("%d",&k);
    
    h =(xn-x0)/k;
    
    fx0 = f(x0);
    fxn = f(xn);
    
    term = f(x0)+f(xn);
    
    for(i=1;i<=k-1;i++){
        if(i%3!=0){
            a= x0+i*h;
            term = term+3*f(a);
        }
        
        else
        {
            a= x0+i*h;
            term = term+2*f(a);
            
        }
    }
    
    v= 3/8.0 *h*term;
    
    printf("value of  integration = %f",v);
    
}


#include <stdio.h>

// Define the function to integrate (example: f(x) = x^2)
double f(double x) {
    return x * x;  // Example function: f(x) = x^2
}

// Trapezoidal Rule to calculate the integral
double trapezoidalRule(double a, double b, int n) {
    double h = (b - a) / n, sum = (f(a) + f(b)) / 2;
    for (int i = 1; i < n; i++) sum += f(a + i * h);
    return sum * h;
}

int main() {
    printf("Sharad Bista\n");
    double a, b;
    int n;
    
    // Input limits and subintervals
    printf("Enter limits a, b and subintervals n: ");
    scanf("%lf %lf %d", &a, &b, &n);
    
    // Output the result
    printf("Integral: %.6f\n", trapezoidalRule(a, b, n));
    return 0;
}

#include <stdio.h>

// Define the function to integrate (example: f(x) = x^2)
double f(double x) {
    return x * x;  // Example function: f(x) = x^2
}

// Simpson's 1/3 Rule to calculate the integral
double simpsonsRule(double a, double b, int n) {
    double h = (b - a) / n, sum = f(a) + f(b);
    for (int i = 1; i < n; i++) {
        if (i % 2 == 0)
            sum += 2 * f(a + i * h);  // Even index
        else
            sum += 4 * f(a + i * h);  // Odd index
    }
    return sum * h / 3;
}

int main() {
    printf("Sharad Bista\n");
    double a, b;
    int n;
    
    // Input limits and subintervals (n must be even)
    printf("Enter limits a, b and subintervals n (even): ");
    scanf("%lf %lf %d", &a, &b, &n);
    
    // Output the result
    printf("Integral: %.6f\n", simpsonsRule(a, b, n));
    return 0;
}

#include <stdio.h>

// Define the function to integrate (example: f(x) = x^2)
double f(double x) {
    return x * x;  // Example function: f(x) = x^2
}

// Simpson's 3/8 Rule to calculate the integral
double simpsons38Rule(double a, double b, int n) {
    if (n % 3 != 0) {
        printf("n must be a multiple of 3.\n");
        return -1; // Return an error if n is not a multiple of 3
    }

    double h = (b - a) / n;
    double sum = f(a) + f(b);

    for (int i = 1; i < n; i++) {
        if (i % 3 == 0) {
            sum += 2 * f(a + i * h); // Every third term is multiplied by 2
        } else {
            sum += 3 * f(a + i * h); // Other terms are multiplied by 3
        }
    }

    return (3 * h / 8) * sum;
}

int main() {
    printf("Sharad Bista\n");
    double a, b;
    int n;
    
    // Input limits and subintervals (n must be multiple of 3)
    printf("Enter limits a, b and subintervals n (multiple of 3): ");
    scanf("%lf %lf %d", &a, &b, &n);
    
    // Output the result
    double result = simpsons38Rule(a, b, n);
    if (result != -1) {
        printf("Integral: %.6f\n", result);
    }

    return 0;
}








// Gausssian quadrature

#include<stdio.h>
#include<math.h>
float f(float x){
    return  x*x*x+1;
    
}
int main(void){
    printf("sharad  bista\n  ");
    float c1,c2,z1,z2,b,a,x1,x2,v;
    printf("Enter the lower and upper limit\n");
    scanf("%f%f",&a,&b);
    
    
    // setting the value of the parameters
    c1=c2=1;
    z1 =-0.557735;
    z2=0.57735;
    
    //calculating xi
    
    x1=(b-a)/2*z1+(b+a)/2;
    x2=(b-a)/2*z2+(b+a)/2;
    
    //calculating integral value
    v=(b-a)/2*((f(x1))+(f(x2)));
    
    printf("value integration is =%f",v);
    
}


//Romberg estimate
#include<stdio.h>
#include<math.h>
float f(float x){
    return  x*x*x+1;
    
}
int main(void){
    printf("Sharad bista\n");
    float x0,xn,T[10][10],h,sm,sl,a;
    int i,k,c,r,m,p,q;
    
    printf("Enter the lower and upper limit\n");
    scanf("%f%f",&x0,&xn);
    
    printf("Enter p & q of required T(p,q)\n");
    scanf("%d%d",&p,&q);
    
    h = xn-x0;
    
    T[0][0] = h/2*((f(x0))+(f(xn)));
    
    for(i=1;i<=p;i++){
        
        sl = pow(2,i-1);
        sm =0;
        for(k=1;k<=sl;k++){
            a =x0 +(2*k-1)*h/pow(2,i);
            sm = sm+(f(a));
            
        }
        T[i][0]=T[i-1][0]/2+sm*h/pow(2,i);
    }
    
    for(c=1;c<=p;c++){
        for(k=1;k<=c&&k<=q;k++){
            m = c-k;
            T[m+k][k]=(pow(4,k)*T[m+k][k-1]-T[m+k-1][k-1])/(pow(4,k)-1);
        }
        
    }
                
    printf("Romberge Estimate of integration is = %f",T[p][q]);
    
}

//Taylor series
#include<stdio.h>
#include<math.h>
float fact(int n){
    if(n==1)
        return 1;
    else{
        return (n*fact(n-1));
    }
    
    
}
int main(void){
printf("sajan bista\n Taylor series\n");
    
float x,x0,yx0,yx,fdy,sdy,tdy;
    
printf("Enter initial values of x & y \n");
scanf(" %f%f",&x0,&yx0);
    
printf("Enter x at which function to be evaluated \n");
scanf("%f",&x);
    
fdy=(x0)*(x0)+(yx0)*(yx0);//First Derivative
    
sdy= 2*(x0) + 2*(yx0)*fdy;// Second Derivative
    
tdy=2+2*yx0*sdy+2*fdy*fdy;// Third Derivative
    
yx=yx0+(x-x0)*fdy+(x-x0)*(x-x0)*sdy /fact(2)+(x-x0)*(x-x0)*(x-x0)*tdy /fact(3);
    
printf("Function value at x=%f is %f) n",x,yx);
           
    }

//Euler's method
#include<stdio.h>
#include<math.h>

float f(float x,float y){
    return 2*y/x;
}


int main(void){
printf("Sharad Bista\n");
float x, xp, x0,y0,y,h;
    
printf("Enter initial values of x & y \n");
scanf("%f%f", &x0,&y0);

printf("Enter x at which function to be Evaluated \n");
scanf("%f", &xp);
    
printf("Enter the step size\n ");
scanf("%f", &h);
    
y=y0;
x=x0;
    
for(x=x0;x<xp;x=x+h){
    y=y+f(x,y)*h;
}

printf("Function value at x=%f is %f\n",xp,y);
}
 

#include<stdio.h>

#include<math.h>

float f(float x,float y) {
    return 2*(y)/(x);
}
int main(void){
    printf(" Heun's method\n");
    printf("sharad bista\n");
    float x, xp, x0, y0, y, h, m1, m2;
    printf("Enter initial values of x & y \n");
    scanf("%f%f", &x0, &y0);
    printf("Enter x at which function to be Evaluated \n");
    scanf("%f", &xp);
    printf("Enter the step size\n");
    scanf("%f", &h);
    y = y0;
    x = x0;
    for(x=x0;x<xp;x=x+h){
        m1=f(x,y);
        m2=f(x+h,y+h*m1);
        y=y+h/2*(m1+m2);
    }
    
    printf("Function value at x=%f is %f\n",xp,y);
}




#include<stdio.h>

#include<math.h>

float f(float x,float y) {
    return 2*(x)+(y);
}
int main(void){
    
    printf("Fourth Order Runge-Kutta Method\n");
    printf("sharad bista \n ");
    float x,xp,x0,y0,y,h,m1,m2,m3,m4;
    printf("Enter initial values of x & y \n");
    scanf("%f%f",&x0,&y0);
    printf("Enter x at which function to be Evaluated\n");
    scanf("%f",&xp);
    printf("Enter the step size\n");
    scanf("%f",&h);
    y=y0;
    x=x0;
    for(x=x0;x<xp;x=x+h){
        m1=f(x,y);
        m2=f(x+1/2.0*h,y+1/2.0*h*m1);
        m3=f(x+1/2.0*h,y +1/2.0*h*m2);
        m4=f(x+h,y +h*m3);
        y=y+h/6*(m1+2*m2+2*m3+m4);
    }
    printf("Function value at x=%f is %f\n",xp,y);
    
    
}


//shooting method
#include <stdio.h>
#include <math.h>

#define f1(x, y, z) (z)          // Define the first function
#define f2(x, y, z) (6 * (x))    // Define the second function

int main(void)
{
    printf("sharad bista\n");
    float xa, xb, ya, yb, x, y, z, xp, h, sol, ny, nz, error, E, g[3], v[3], gs;
    int i;

    printf("Enter Boundary Conditions (xa, ya, xb, yb):\n");
    scanf("%f %f %f %f", &xa, &ya, &xb, &yb);

    printf("Enter x at which value is required:\n");
    scanf("%f", &xp);

    printf("Enter the step size:\n");
    scanf("%f", &h);

    printf("Enter accuracy limit:\n");
    scanf("%f", &E);

    x = xa;
    y = ya;
    g[1] = z = (yb - ya) / (xb - xa); // Initial slope guess
    printf("Initial slope (g[1]) = %f\n", g[1]);

    // First shooting iteration
    while (x < xb) {
        ny = y + (f1(x, y, z)) * h;
        nz = z + (f2(x, y, z)) * h;
        x += h;
        y = ny;
        z = nz;
        if (fabs(x - xp) < 1e-6) { // Check if we reach xp
            sol = y;
        }
    }

    v[1] = y;
    if (y < yb) {
        g[2] = z = 2 * g[1];
    } else {
        g[2] = z = 0.5 * g[1];
    }
    printf("Updated slope guess (g[2]) = %f\n", g[2]);

    // Second shooting iteration
    x = xa;
    y = ya;
    z = g[2];
    while (x < xb) {
        ny = y + (f1(x, y, z)) * h;
        nz = z + (f2(x, y, z)) * h;
        x += h;
        y = ny;
        z = nz;
        if (fabs(x - xp) < 1e-6) {
            sol = y;
        }
    }
    v[2] = y;

    // Iterative correction
    while (1) {
        x = xa;
        y = ya;
        gs = g[2] - (v[2] - yb) / (v[2] - v[1]) * (g[2] - g[1]);
        z = gs;

        while (x < xb) {
            ny = y + (f1(x, y, z)) * h;
            nz = z + (f2(x, y, z)) * h;
            x += h;
            y = ny;
            z = nz;
            if (fabs(x - xp) < 1e-6) {
                sol = y;
            }
        }

        error = fabs(y - yb) / yb;
        v[1] = v[2];
        v[2] = y;
        g[1] = g[2];
        g[2] = gs;

        if (error < E) {
            printf("y(%f) = %f\n", xp,sol);
            break;
        }
    }

    return 0;
}


#include <stdio.h>
#include <math.h>
#define MAX 10 // Maximum dimension of the grid
#define EPSILON 1e-6 // Convergence criteria

int main(void){
    printf("sharad bista\n");
    int n, i, j, iteration = 0;
    float tl, tr, tb, tu;  // Temperatures on the boundaries
    float grid[MAX][MAX], newGrid[MAX][MAX];
    float error, maxError;

    // Input the dimension of the plate
    printf("Enter the number of grid points per side (n x n): ");
    scanf("%d", &n);

    if (n >= MAX) {
        printf("Error: n exceeds maximum supported size (%d).\n", MAX - 1);
        return 1;
    }

    // Input boundary conditions
    printf("Enter the temperature at the left boundary: ");
    scanf("%f", &tl);
    printf("Enter the temperature at the right boundary: ");
    scanf("%f", &tr);
    printf("Enter the temperature at the bottom boundary: ");
    scanf("%f", &tb);
    printf("Enter the temperature at the top boundary: ");
    scanf("%f", &tu);

    // Initialize the grid with boundary conditions
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == 0)              // Bottom boundary
                grid[i][j] = tb;
            else if (i == n - 1)     // Top boundary
                grid[i][j] = tu;
            else if (j == 0)         // Left boundary
                grid[i][j] = tl;
            else if (j == n - 1)     // Right boundary
                grid[i][j] = tr;
            else                     // Interior points
                grid[i][j] = 0.0;
        }
    }

    // Iteratively solve using Gauss-Seidel method
    do {
        maxError = 0.0;
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                // Update the grid using the finite difference formula
                newGrid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] +
                                       grid[i][j-1] + grid[i][j+1]);

                // Compute the error
                error = fabs(newGrid[i][j] - grid[i][j]);
                if (error > maxError)
                    maxError = error;

                // Update the grid in place
                grid[i][j] = newGrid[i][j];
            }
        }
        iteration++;
    } while (maxError > EPSILON);

    // Output the result
    printf("\nSolution converged in %d iterations.\n", iteration);
    printf("Temperature distribution on the plate:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%8.4f ", grid[i][j]);
        }
        printf("\n");
    }

    return 0;
}

*/
#include <stdio.h>
#include <math.h>

#define MAX 50   // Maximum grid size
#define EPSILON 1e-6 // Convergence criterion

int main() {
    printf("sharad bista\n");
    int n, i, j, iter = 0;
    double h, x[MAX][MAX], f[MAX][MAX], error, maxError;

    // Input grid size and initialize
    printf("Enter grid size (n x n): ");
    scanf("%d", &n);

    if (n >= MAX) {
        printf("Grid size too large! Max is %d.\n", MAX - 1);
        return 1;
    }

    printf("Enter grid spacing (h): ");
    scanf("%lf", &h);

    // Initialize source term f(x, y) and grid x[i][j]
    printf("Enter the source term values f(x, y):\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("f[%d][%d]: ", i, j);
            scanf("%lf", &f[i][j]);
            x[i][j] = 0.0; // Initial guess
        }
    }

    // Iterative Gauss-Seidel method
    do {
        maxError = 0.0;
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                double oldVal = x[i][j];
                x[i][j] = 0.25 * (x[i - 1][j] + x[i + 1][j] +
                                  x[i][j - 1] + x[i][j + 1] -
                                  h * h * f[i][j]);
                error = fabs(x[i][j] - oldVal);
                if (error > maxError)
                    maxError = error;
            }
        }
        iter++;
    } while (maxError > EPSILON);

    // Output solution
    printf("\nSolution converged in %d iterations.\n", iter);
    printf("Solution:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%8.4f ", x[i][j]);
        }
        printf("\n");
    }

    return 0;
}
/*
//linear regression
#include <stdio.h>

int main() {
    printf("sajan bista\n");
    int n, i;
    float a = 0, b = 0, x[10], y[10];
    float sx = 0, sy = 0, sxy = 0, sx2 = 0;

    // Input the number of data points
    printf("Enter the number of points n: ");
    scanf("%d", &n);

    // Input the x and y values
    printf("Enter the values of x and y:\n");
    for (i = 0; i < n; i++) {
        printf("x[%d], y[%d]: ", i, i);
        scanf("%f %f", &x[i], &y[i]);
    }

    // Calculate summations
    for (i = 0; i < n; i++) {
        sx += x[i];
        sy += y[i];
        sxy += x[i] * y[i];
        sx2 += x[i] * x[i];
    }

    // Compute the slope (b) and intercept (a)
    b = ((n * sxy) - (sx * sy)) / ((n * sx2) - (sx * sx));
    a = (sy / n) - (b * (sx / n));

    // Display the fitted line equation
    printf("\nFitted line equation: y = %.2f + %.2f * x\n", a, b);

    return 0;
}

//exponential
#include <stdio.h>
#include <math.h> // For log() and exp()

int main() {
    printf("sajan bista\n");
    int n, i;
    float x[10], y[10], log_y[10];
    float sx = 0, sy = 0, sxy = 0, sx2 = 0;
    float a, b, A;

    // Input the number of data points
    printf("Enter the number of points n: ");
    scanf("%d", &n);

    // Input the x and y values
    printf("Enter the values of x and y:\n");
    for (i = 0; i < n; i++) {
        printf("x[%d], y[%d]: ", i, i);
        scanf("%f %f", &x[i], &y[i]);

        if (y[i] <= 0) {
            printf("Error: y values must be positive for exponential regression.\n");
            return 1;
        }
        log_y[i] = log(y[i]); // Compute ln(y)
    }

    // Calculate summations
    for (i = 0; i < n; i++) {
        sx += x[i];
        sy += log_y[i];
        sxy += x[i] * log_y[i];
        sx2 += x[i] * x[i];
    }

    // Compute b and A
    b = ((n * sxy) - (sx * sy)) / ((n * sx2) - (sx * sx));
    A = (sy / n) - (b * (sx / n));

    // Compute a = exp(A)
    a = exp(A);

    // Display the fitted exponential equation
    printf("\nFitted exponential equation: y = %.2f * e^(%.2f * x)\n", a, b);

    return 0;
}

//polynomial regression
#include <stdio.h>
#include <math.h>

#define MAX 10 // Maximum number of data points
#define DEG 5  // Maximum degree of the polynomial

int main() {
    printf("sajan bista\n");
    int n, degree, i, j, k;
    double x[MAX], y[MAX], X[2 * DEG + 1], B[DEG + 1], A[DEG + 1][DEG + 2], coeff[DEG + 1];

    // Input the number of data points
    printf("Enter the number of data points (n): ");
    scanf("%d", &n);

    // Input the degree of the polynomial
    printf("Enter the degree of the polynomial: ");
    scanf("%d", &degree);

    // Input x and y values
    printf("Enter the values of x and y:\n");
    for (i = 0; i < n; i++) {
        printf("x[%d], y[%d]: ", i, i);
        scanf("%lf %lf", &x[i], &y[i]);
    }

    // Initialize summations for X and B
    for (i = 0; i <= 2 * degree; i++) {
        X[i] = 0;
        for (j = 0; j < n; j++) {
            X[i] += pow(x[j], i);
        }
    }

    for (i = 0; i <= degree; i++) {
        B[i] = 0;
        for (j = 0; j < n; j++) {
            B[i] += pow(x[j], i) * y[j];
        }
    }

    // Construct the augmented matrix
    for (i = 0; i <= degree; i++) {
        for (j = 0; j <= degree; j++) {
            A[i][j] = X[i + j];
        }
        A[i][degree + 1] = B[i];
    }

    // Perform Gaussian elimination
    for (i = 0; i <= degree; i++) {
        for (j = 0; j <= degree; j++) {
            if (j != i) {
                double ratio = A[j][i] / A[i][i];
                for (k = 0; k <= degree + 1; k++) {
                    A[j][k] -= ratio * A[i][k];
                }
            }
        }
    }

    // Extract coefficients
    for (i = 0; i <= degree; i++) {
        coeff[i] = A[i][degree + 1] / A[i][i];
    }

    // Display the polynomial equation
    printf("\nThe fitted polynomial is:\n");
    printf("y = ");
    for (i = 0; i <= degree; i++) {
        if (i == 0) {
            printf("%.4lf", coeff[i]);
        } else {
            printf(" + %.4lf*x^%d", coeff[i], i);
        }
    }
    printf("\n");

    return 0;
}


//gauss elimination method
#include <stdio.h>
#include <math.h>

#define MAX 10 // Maximum number of variables

void gaussElimination(float a[MAX][MAX], int n) {
    int i, j, k;
    float factor, sum, x[MAX];

    // Forward Elimination
    for (k = 0; k < n - 1; k++) {
        for (i = k + 1; i < n; i++) {
            factor = a[i][k] / a[k][k];
            for (j = k; j <= n; j++) {
                a[i][j] -= factor * a[k][j];
            }
        }
    }

    // Back Substitution
    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1];
    for (i = n - 2; i >= 0; i--) {
        sum = 0;
        for (j = i + 1; j < n; j++) {
            sum += a[i][j] * x[j];
        }
        x[i] = (a[i][n] - sum) / a[i][i];
    }

    // Display the solution
    printf("\nThe solution is:\n");
    for (i = 0; i < n; i++) {
        printf("x[%d] = %.4f\n", i + 1, x[i]);
    }
}

int main() {
    printf("sajan bista\n");
    int n, i, j;
    float a[MAX][MAX];

    // Input number of variables
    printf("Enter the number of variables: ");
    scanf("%d", &n);

    // Input augmented matrix
    printf("Enter the augmented matrix (coefficients and constants):\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j <= n; j++) {
            printf("a[%d][%d]: ", i + 1, j + 1);
            scanf("%f", &a[i][j]);
        }
    }

    // Perform Gauss Elimination
    gaussElimination(a, n);

    return 0;
}

#include <stdio.h>
#include <math.h>

#define MAX 10 // Maximum number of variables

void gaussJordan(float a[MAX][MAX], int n) {
    int i, j, k;
    float factor;

    // Convert matrix to reduced row-echelon form
    for (i = 0; i < n; i++) {
        // Make the diagonal element 1
        factor = a[i][i];
        for (j = 0; j <= n; j++) {
            a[i][j] /= factor;
        }

        // Make all other elements in the column 0
        for (k = 0; k < n; k++) {
            if (k != i) {
                factor = a[k][i];
                for (j = 0; j <= n; j++) {
                    a[k][j] -= factor * a[i][j];
                }
            }
        }
    }

    // Display the solution
    printf("\nThe solution is:\n");
    for (i = 0; i < n; i++) {
        printf("x[%d] = %.4f\n", i + 1, a[i][n]);
    }
}

int main() {
    printf("sajan bista\n");
    int n, i, j;
    float a[MAX][MAX];

    // Input number of variables
    printf("Enter the number of variables: ");
    scanf("%d", &n);

    // Input augmented matrix
    printf("Enter the augmented matrix (coefficients and constants):\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j <= n; j++) {
            printf("a[%d][%d]: ", i + 1, j + 1);
            scanf("%f", &a[i][j]);
        }
    }

    // Perform Gauss-Jordan Elimination
    gaussJordan(a, n);

    return 0;
}
//
#include <stdio.h>
#include <math.h>

#define MAX 10  // Maximum size of the matrix

void gaussJordanInverse(float a[MAX][MAX], float inverse[MAX][MAX], int n) {
    int i, j, k;
    float factor;

    // Augmenting the matrix with the identity matrix
    float augmented[MAX][2*MAX];
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            augmented[i][j] = a[i][j];
            augmented[i][j + n] = (i == j) ? 1 : 0;  // Identity matrix
        }
    }

    // Perform Gauss-Jordan elimination
    for (i = 0; i < n; i++) {
        // Make the diagonal element 1
        factor = augmented[i][i];
        for (j = 0; j < 2*n; j++) {
            augmented[i][j] /= factor;
        }

        // Make the rest of the column elements 0
        for (k = 0; k < n; k++) {
            if (k != i) {
                factor = augmented[k][i];
                for (j = 0; j < 2*n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            inverse[i][j] = augmented[i][j + n];
        }
    }
}

int main() {
    printf("sajan bista\n");
    int n, i, j;
    float a[MAX][MAX], inverse[MAX][MAX];

    // Input number of variables (order of the matrix)
    printf("Enter the order of the matrix: ");
    scanf("%d", &n);

    // Input matrix
    printf("Enter the elements of the matrix:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("a[%d][%d]: ", i + 1, j + 1);
            scanf("%f", &a[i][j]);
        }
    }

    // Call Gauss-Jordan method to compute the inverse
    gaussJordanInverse(a, inverse, n);

    // Output the inverse matrix
    printf("\nThe inverse of the matrix is:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%.4f ", inverse[i][j]);
        }
        printf("\n");
    }

    return 0;
}
//matrix factorization method

#include <stdio.h>
#include <stdlib.h>

#define MAX 10  // Maximum size of matrix

// Function to perform LU Decomposition
void luDecomposition(float mat[MAX][MAX], float L[MAX][MAX], float U[MAX][MAX], int n) {
    int i, j, k;

    // Initialize L and U matrices to zero
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            L[i][j] = 0;
            U[i][j] = 0;
        }
    }

    // LU Decomposition
    for (i = 0; i < n; i++) {
        // Upper Triangular Matrix
        for (j = i; j < n; j++) {
            U[i][j] = mat[i][j];
            for (k = 0; k < i; k++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }

        // Lower Triangular Matrix
        for (j = i; j < n; j++) {
            if (i == j) {
                L[i][i] = 1;  // Diagonal elements of L are set to 1
            } else {
                L[j][i] = mat[j][i];
                for (k = 0; k < i; k++) {
                    L[j][i] -= L[j][k] * U[k][i];
                }
                L[j][i] /= U[i][i];
            }
        }
    }
}

// Function to print a matrix
void printMatrix(float mat[MAX][MAX], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", mat[i][j]);
        }
        printf("\n");
    }
}

int main() {
    printf("sajan bista");
    int n, i, j;
    float mat[MAX][MAX], L[MAX][MAX], U[MAX][MAX];

    // Input the order of the matrix
    printf("Enter the order of the matrix: ");
    scanf("%d", &n);

    // Input matrix elements
    printf("Enter the elements of the matrix:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("a[%d][%d]: ", i + 1, j + 1);
            scanf("%f", &mat[i][j]);
        }
    }

    // Perform LU Decomposition
    luDecomposition(mat, L, U, n);

    // Output the lower triangular matrix L
    printf("\nLower Triangular Matrix L:\n");
    printMatrix(L, n);

    // Output the upper triangular matrix U
    printf("\nUpper Triangular Matrix U:\n");
    printMatrix(U, n);

    return 0;
}



//jacobii
#include <stdio.h>
#include <math.h>

#define MAX 10  // Maximum size of the system
#define TOLERANCE 1e-6  // Accuracy tolerance for convergence

// Function to perform Jacobi iteration
void jacobi(float A[MAX][MAX], float b[MAX], float x[MAX], int n, int maxIterations) {
    float x_new[MAX];  // To store the updated solution
    int i, j, k;
    float sum, error;

    // Start iterating
    for (k = 0; k < maxIterations; k++) {
        // Update each variable
        for (i = 0; i < n; i++) {
            sum = b[i];
            for (j = 0; j < n; j++) {
                if (i != j) {
                    sum -= A[i][j] * x[j];
                }
            }
            x_new[i] = sum / A[i][i];
        }

        // Calculate error and check for convergence
        error = 0.0;
        for (i = 0; i < n; i++) {
            error += fabs(x_new[i] - x[i]);
        }

        // Update x to the new values
        for (i = 0; i < n; i++) {
            x[i] = x_new[i];
        }

        // If the error is less than tolerance, the solution has converged
        if (error < TOLERANCE) {
            printf("Converged after %d iterations.\n", k + 1);
            return;
        }
    }

    printf("Maximum iterations reached.\n");
}

// Function to print the solution vector
void printSolution(float x[MAX], int n) {
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %.6f\n", i + 1, x[i]);
    }
}

int main() {
    printf("sajan bista\n");
    int n, i, j, maxIterations;
    float A[MAX][MAX], b[MAX], x[MAX] = {0};  // x is initialized to 0 (initial guess)

    // Input the order of the matrix (n x n)
    printf("Enter the number of variables (n): ");
    scanf("%d", &n);

    // Input matrix A and vector b
    printf("Enter the elements of matrix A (row by row):\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("A[%d][%d]: ", i + 1, j + 1);
            scanf("%f", &A[i][j]);
        }
    }

    printf("Enter the elements of the constant vector b:\n");
    for (i = 0; i < n; i++) {
        printf("b[%d]: ", i + 1);
        scanf("%f", &b[i]);
    }

    // Input maximum iterations
    printf("Enter the maximum number of iterations: ");
    scanf("%d", &maxIterations);

    // Perform Jacobi Iteration
    jacobi(A, b, x, n, maxIterations);

    // Output the solution
    printf("\nSolution vector:\n");
    printSolution(x, n);

    return 0;
}



//gauss sedai
#include <stdio.h>
#include <math.h>

#define MAX 10  // Maximum size of the system
#define TOLERANCE 1e-6  // Accuracy tolerance for convergence

// Function to perform Gauss-Seidel iteration
void gaussSeidel(float A[MAX][MAX], float b[MAX], float x[MAX], int n, int maxIterations) {
    float sum;
    int i, j, k;
    float error;

    // Start iterating
    for (k = 0; k < maxIterations; k++) {
        // Update each variable
        for (i = 0; i < n; i++) {
            sum = b[i];
            for (j = 0; j < n; j++) {
                if (i != j) {
                    sum -= A[i][j] * x[j];
                }
            }
            x[i] = sum / A[i][i];
        }

        // Calculate error and check for convergence
        error = 0.0;
        for (i = 0; i < n; i++) {
            error += fabs(x[i] - b[i]);
        }

        // If the error is less than tolerance, the solution has converged
        if (error < TOLERANCE) {
            printf("Converged after %d iterations.\n", k + 1);
            return;
        }
    }

    printf("Maximum iterations reached.\n");
}

// Function to print the solution vector
void printSolution(float x[MAX], int n) {
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %.6f\n", i + 1, x[i]);
    }
}

int main() {
    printf("sajan bista");
    int n, i, j, maxIterations;
    float A[MAX][MAX], b[MAX], x[MAX] = {0};  // x is initialized to 0 (initial guess)

    // Input the order of the matrix (n x n)
    printf("Enter the number of variables (n): ");
    scanf("%d", &n);

    // Input matrix A and vector b
    printf("Enter the elements of matrix A (row by row):\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("A[%d][%d]: ", i + 1, j + 1);
            scanf("%f", &A[i][j]);
        }
    }

    printf("Enter the elements of the constant vector b:\n");
    for (i = 0; i < n; i++) {
        printf("b[%d]: ", i + 1);
        scanf("%f", &b[i]);
    }

    // Input maximum iterations
    printf("Enter the maximum number of iterations: ");
    scanf("%d", &maxIterations);

    // Perform Gauss-Seidel Iteration
    gaussSeidel(A, b, x, n, maxIterations);

    // Output the solution
    printf("\nSolution vector:\n");
    printSolution(x, n);

    return 0;
}
 */

