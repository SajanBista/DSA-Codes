//simpson's 1/3 rule

#include<stdio.h>
#define f(x) 4*x*x+2*x-5
int main(void){
    float x0,x1,x2,h,fx0,fx1,fx2,val;
    printf("Sajan Bista\n");
    printf("enter the upper limit and lower limit \t");
    scanf("%f,%f",&x0,&x2);
    h=(x2-x0)/2;
    x1 = x0+h;
    fx0=f(x0);
    fx1=f(x1);
    fx2=f(x2);
    
    val = h/3*(4*f(x1)+f(x0)+f(x2));
    printf("the answer of above given question is %f",&val);
    
    
}


