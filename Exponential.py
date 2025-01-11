"""sum = 0
for i in range(0,50):
    sum+=i
print(sum)

reverse = []
for i in range(10,0,-1):
    reverse.append(i)

print(reverse)

#multipication table 
for i in range(1,11):
    for j in range(1,11):
        print(i*j, end=" ")
        print("\n")"""

"""List iteration
Iterate over the list [2, 4, 6, 8, 10] and print each element.

Sum of list elements
Find the sum of all elements in a list: [1, 3, 5, 7, 9].

Factorial
Write a program to calculate the factorial of a number using a for loop.

Count vowels in a string
Given a string, count the number of vowels in it. For example, in "Hello World," there are 3 vowels.
"""
"""list =[2, 4, 6, 8, 10]
sum = 0
for i in list:
    sum+=i
    print(i)
print("the sum is\t",sum)
fruits = ["apple", "banana", "cherry","orange","grapes"]
for i in range(len(fruits)):
    print(fruits[i])

#Factorial
fact = 1
num =int(input("Enter the number"))
for i in range(num,0,-1):
    fact = fact*i
    print(fact)
print(fact)
"""
"""string = "Hello World"
vowels = 0
for i in string:
    if i in 'aeiou':
        vowels+=1
        print(i)
print(vowels)"""

for i in range(1, 50):
    if i > 1:
        for j in range(2, int(i**0.5) + 1):
            if (i % j) == 0:
                break
        else:
            print(i)