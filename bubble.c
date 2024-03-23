/*sorting programms*/
/*1. for bubble sorting*/
#include<stdio.h>
#include<conio.h>
void bubble(int A[],int n);
int main()
{
int n;
int A[]={10,20,12,8,1,0};
n= sizeof(A)/sizeof(A[0]);
//printf("the elements before sorting are %d",A[]);

bubble(A,n);
printf("sorted array:\n");
int i;
for (i=0 ;i<n;i++)
printf("%d\n",A[i]);
printf("\n");

return 0;

           
           
}
void bubble(int A[],int n)
{
     int temp,i,j;
for(i=0;i<n-1;i++)
{  
for(j=0;j<n-1-i;j++)
 {if(A[j]>A[j+1])
 {
                 temp=A[j];
 A[j]=A[j+1];
 A[j+1]=temp;
}
}
  }
}

           


