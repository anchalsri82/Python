import math
f=open("D:\Projects\Python\Practice\Practice3_File_Humpty.txt")
rhyme = f.read()
print(rhyme)

## above code does close file connection.
f.close()

## to avoid file close issue use with, similiar to using in c#
with open("D:\Projects\Python\Practice\Practice3_File_Humpty.txt") as f:
    rhyme2 = f.read()
print(rhyme)


f = open ("D:\Projects\Python\Practice\Practice3_File_Humpty.txt", "w") # use "a" to apend
pi = 3.14
message = "Total:" + str(1025.5) + str(pi==2)
  
#print >> f, "Total:", 1025.5, pi==2 # >> isn't supported in Python 3
f.write (message)
f.close()


print(2**100)
print(4+8/2)
print(pi)
#print(e)#
x=2
print(x**4) # 2 to the power 4
print(math.pi)
print(math.e)

