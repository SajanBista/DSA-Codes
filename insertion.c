#include<stdio.h>
void insertion(int[],int);
int main()
{
    int arr[100],n,i;
    printf("enter the number of term you want to sort.");
    scanf("%d",&n);
    printf("enter the numbers");
    for(i=0;i<n;i++)
       {
        scanf("%d",&arr[i]);
    }
    printf(" the number before sorting are.");
    for(i=0;i<n;i++){
        printf("%d\t",arr[i]);
    }
    insertion(arr,n);
    printf("\nthe elements after sorting using insertion sort is");
    for(i=0;i<n;i++){
        printf("%d\t",arr[i]);
    }
}

void insertion(int arr[],int n)
{
    int i,temp,j;
    for(i=0;i<n;i++){
        temp=arr[i];
        j=i-1;
        while(j>0 && arr[j]>temp){
            arr[j+1]=arr[j];
            j--;
            
            
        }
        arr[j+1]=temp;
    }
    
}