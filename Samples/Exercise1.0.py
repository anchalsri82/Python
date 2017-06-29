#Exercise 1.0

#1
print('''coffee
cafe\'
caffe\'
kaffee''')

#2
print ('''It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.''')

#3
cars = 100
space_in_a_car = 4.0
drivers = 30
passengers = 90
cars_not_driven = cars - drivers
car_driven =drivers
carpool_capacity = car_driven*space_in_a_car
average_passenger_per_car = passengers / car_driven

print("There are", cars, "cars available.")
print("There are only ", drivers, " drivers available")
print("There will be ", cars_not_driven, "empty cars today.")
print("We can transport ", carpool_capacity, " people today.")
print("We have ", passengers," to carpool today.")
print("We need to put about ", average_passenger_per_car, " in each car.")