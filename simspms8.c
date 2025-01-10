// 3/8 rule of simpson's

#include<stdio.h>
#define f(x) 3*x*x*x +2

int main(void){
    printf("Sajan Bista");
    float x1,x2,x3,x0,h,val,fx1,fx2,fx0,fx3;
    printf("enter the upper and lower bound");
    scanf("%f,%f",&x0,&x3);
    
    h = (x3-x0)/3;
    x1 = x0 + h;
    x2 = x0 +2*h;
    fx0 =f(x0);
    fx1 =f(x1);
    fx2 =f(x2);
    fx3 =f(x3);
    
    val = 3*h/8*(fx0 +3*fx1+3*fx2+fx3);
    
    printf(" the result of your input for simpson's 3/8 rule is %f",&val);
    
    
}
