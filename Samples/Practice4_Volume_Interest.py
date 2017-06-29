import math
#V = 1/3 * pi * r **2 * h

print("Enter Radius:")
radius = input()
print("Enter Height:")
height = input()
volume = 1 / 3 * math.pi * float(radius) * float(radius) * float(height)
print("Volume = " + str(volume))


# t = 0  Me^(r(T-t)) T = 15 - Continuous
# t = 0  M(1+rT) = 15 - Discrete

Principle = 500
Time = 1.5
Rate = 3.5
Maturity = Principle * (1 + Rate * Time)

print("Maturity = " + str(Maturity))