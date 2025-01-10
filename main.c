#include <stdio.h>
#include <math.h>

#define a(x) (a3*x*x*x + a2*x*x + a1*x + a0) / (-a1)

float a0, a1, a2, a3;

int main(void) {
    printf("sajan Bista\n fixed point\n");
    float x0, x1, e, er;

    printf("Enter coefficients a3, a2, a1, and a0: ");
    scanf("%f %f %f %f", &a3, &a2, &a1, &a0);

    printf("Enter initial guess and E: ");
    scanf("%f %f", &x0, &e);

    while (1) {
        x1 = a(x0);
        er = fabs((x1 - x0) / x1);

        if (er < e) {
            printf("Root = %f\n", x1);
            break;
        }

        x0 = x1;
    }

    return 0;
}
