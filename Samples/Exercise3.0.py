#PG50
import math
# 1.Fahrenheit to Celcius => 5(F-32)= 9C

print("Enter Temparature in Fahrenheit:")
F=float(input())
C = 5 * (F-32) / 9

print( "temperature in Fahrenheit %s Celcius %s" % (C, F))


# 2. Time period of pendulum given length of string L and 
# constant G = 2pi((L/G)**(1/2))
print("Enter Lenght of Pendulum :")
length=float(input())
G=9.81
Time = 2 * math.pi * ((length/G)**0.5)

print(Time)


#3 Calculate the PV of GBP 5400 in three years time given the constant risk-free
# interest rate 3.4%

Principle = 5400
Time = 3
rate = 3.4
PV = Principle * (1 + rate /100 * Time)
print("PV Discreate= %s" % (PV))

#4 How much is GBP 2500 worth in five years time if the constant risk-free 
#interest rate is 4.4 % with continuous compounding.
#
Principle = 5400
Time = 5
rate = 4.4
PV = Principle * (math.e **( rate /100 * Time))
print("PV continuous= %s" % (PV))


#5 Experiment with different mathematical functions from the math module by 
# writing various mathematical scripts


#6 Write a program to check the following formulae by inputting n (your choice)
# and then computing and comparing both sides of the equation:
#a. Summation (i=1 to n) I = n(n+1) / 2
#b. Summation (i=1 to n) I**2 = (n(n+1)(2n+1)) / 6
#c. Summation (i=1 to n) I ** 3 = (n**2 (n+1)**2) / 4

n = int(input())
# a
sumi=0
i=0
for i in range(1,n+1):
    sumi= sumi+i
    
print("Summation (i=1 to n) I = %s \n n(n+1) / 2 = %d" % (sumi, (n*(n+1))/2))



# b
sumi2=0
i=0
for i in range(1,n+1):
    sumi2= sumi2+i**2
    
print("Summation (i=1 to n) I**2 = %s \n (n(n+1)(2n+1))/6 = %d" 
% (sumi2, ((n*(n+1)*(2*n+1)) / 6)))


sumi3=0
# c

i=0
for i in range(1,n+1):
    sumi3= sumi3+i**3

sumi3Math = (n**2 * (n+1)**2) / 4
    
print("Summation (i=1 to n) I**3 = %s \n (n**2 * (n+1)**2) / 4 = %d" 
% (sumi3, sumi3Math))

















