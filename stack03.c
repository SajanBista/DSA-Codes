//creating simple stack.
#include<stdio.h>
#define SIZE 10
int top=-1;
int stack[SIZE];

int isfull(){
if(top==SIZE-1)
return 1;
else
return 0;
} 
int isempty(){
	if(top==-1)
	return 1;
	else
	return 0;
}
void push(int data){
	if(isfull()){
		printf("stack is full");
	}
	else{
	top++;
	stack[top]=data;}
}
void pop(){
	
	int data;
	if(isempty()){
		printf("stack is empty");
	}
	else{
		data=stack[top];
		top--;
		printf("popped element:%d\n",data);
	}
}
void peek(){
	if(isempty()){
		printf("stack is empty");
	}
	else{
		printf("\ntop element:%d\n",stack[top]);
	}
}
void display(){
int i;
	if(isempty()){
printf("stack is empty.");}
else{
	printf("\nDisplaying all elements\n");
	for(i=top;i>=0;i--)
	{
		printf("%d\t",stack[i]);}
	}
}
int main(){
	push(7);
	push(2);
	push(9);
	push(11);
	display();
	peek();
	pop();
	display();
	return 0;
}	
