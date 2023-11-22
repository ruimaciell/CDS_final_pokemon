#Import Packages
import random
import re
import math

#1. In this exercise we will make a "Patient" class
#
# The Patient class should store the state of
# a patient in our hospital system.
#
#
# 1.1)
# Create a class called "Patient".
# The constructor should have two parameters
# (in addition to self, of course):
#
# 1. name (str)
# 2. symptoms (list of str)
#
# the parameters should be stored as attributes
# called "name" and "symptoms" respectively

#1.3)
# Create a method called has_covid()
# which takes no parameters.
#
# "has_covid" returns a float, between 0.0
# and 1.0, which represents the probability
# of the patient to have Covid-19
#
# The probability should work as follows:
#
# 1. If the user has had the test "covid"
#    then it should return .99 if the test
#    is True and 0.01 if the test is False
# 2. Otherwise, probability starts at 0.05
#    and increases by 0.1 for each of the
#    following symptoms:
#    ['fever', 'cough', 'anosmia']


class Patient:
    def __init__(self, name, symptoms):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        if not all(isinstance(symptom, str) for symptom in symptoms):
            raise TypeError("Symptoms must be a list of strings")
        self.name = name
        self.symptoms = symptoms
        self.tests = {}

    def add_test(self, name_of_the_test, results):
        if not isinstance(name_of_the_test, str):
            raise TypeError("Name must be a string")
        if not isinstance(results, bool):
            raise TypeError("Results must be a boolean")
        self.tests[name_of_the_test] = results

    def has_covid(self):
        for test_name, test_result in self.tests.items():
                if re.search(r'covid', test_name, re.IGNORECASE):
                    if test_result == True:
                        return 0.99
                    else:
                        return 0.01
        symptom_count = sum(symptom in self.symptoms for symptom in ['fever', 'cough', 'anosmia'])
        return 0.05 + 0.1 * symptom_count
        
# """Tests that we have done"""

# # patient1 = Patient("John", ["nosmia", "fever"])
# # patient1.add_test("Covid", False)
# # patient1.add_test("Flu", False)
# # patient1.add_test("Covid19", True)
# # patient1.add_test("COVIDfsadas", False)

# # probability = patient1.has_covid()
# # Test the has_covid method
# print(f"The probability of having Covid-19: {probability}")

# 2. In this exercise you will make an English Deck class made of Card classes
# 
# the Card class should represent each of the cards
#
# the Deck class should represent the collection of cards and actions on them

# 2.1) Create a Card class called "Card".
# The constructor (__init__ ) should have two parameters the "suit" and the "value" and the suit of the card.
# The class should store both as attributes.

class Card:
    def __init__(self, suit, value):
        if not isinstance(suit, str):
            raise TypeError("Suit must be a string")
        if not isinstance(value, str):
            raise TypeError("Value must be a string")
        self.suit = suit
        self.value = value

# 2.2) Create a Deck class called "Deck".
# The constructor will create an English Deck (suits: Hearts, Diamonds, Clubs, Spades and values: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K). It will create a list of cards that contain each of the existing cards in an English Deck.
# Create a method called "shuffle" that shuffles the cards randomly. 
# Create a method called "draw" that will draw a single card and print the suit and value. When a card is drawn, the card should be removed from the deck.

class Deck:
    def __init__(self):
        self.Cards = []
        for suit in ["Hearts", "Diamonds", "Clubs", "Spades"]:
            for value in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
                self.Cards.append(Card(suit, value))

    def shuffle(self):
        random.shuffle(self.Cards)

    def draw(self):
        if self.Cards:
            drawn_card = self.Cards.pop()
            print(f"Drawn card: {drawn_card.value} of {drawn_card.suit}")
        else:
            print("The deck is empty.")
    



# """Tests that we have done"""
# deck = Deck()
# deck.shuffle()
# drawn_card = deck.draw()
# deck.print_deck()


# 3. In this exercise you will create an interface that will serve as template 
# for different figures to compute their perimeter and surface. 

# 3.1Create an abstract class (interface) called "PlaneFigure" with two abstract methods:
# compute_perimeter() that will implement the formula to compute the perimiter of the plane figure.
# compute_surface() that will implement the formula to compute the surface of the plane figure.

def check_positive_num(*args):
    for var_value in args:
        if not isinstance(var_value, (int, float)):
            raise TypeError("Variable must be an int or float")
        elif var_value <=0:
            raise ValueError("Variable must be higher than 0")

class PlaneFigure:
    def compute_perimeter(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def compute_surface(self):
        raise NotImplementedError("Subclass must implement abstract method")


# 3.2 Create a child class called "Triangle" that inherits from "PlaneFigure" and has as parameters in the constructor "base", "c1", "c2", "h". ("base" being the base, "c1" and "c2" the other two sides of the triangle and "h" the height). Implement the abstract methods with the formula of the triangle.


class Triangle(PlaneFigure):
    def __init__(self, base, c1, c2, h):
        self.base = base
        self.c1 = c1
        self.c2 = c2
        self.h = h
        check_positive_num(base, c1, c2, h)


    def compute_perimeter_triangle(self):
        return self.base + self.c1 + self.c2

    def compute_surface_triangle(self):
        return 0.5 * self.base * self.h


# 3.3 Create a child class called "Rectangle" that inherits from "PlaneFigure" and has as parameters in the constructor "a", "b" (sides of the rectangle). Implement the abstract methods with the formula of the rectangle.

class Rectangle(PlaneFigure):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        check_positive_num(a,b)

    def compute_perimeter_rectangle(self):
        return 2 * (self.a + self.b)
    def compute_surface_rectangle(self):
        return self.a * self.b


# 3.3 Create a child class called "Circle" that inherits from "PlaneFigure" and has as parameters in the constructor "radius" (radius of the circle). Implement the abstract methods with the formula of the circle.

class Circle(PlaneFigure):
    def __init__(self, radius):
        self.radius = radius
        check_positive_num(radius)
    def compute_perimeter_circle(self):
        return 2 * math.pi * self.radius
    def compute_surface_circle(self):
        return math.pi * self.radius ** 2
    
# """Tests that we have done"""
# triangle1=Triangle(4,3,5,12)
# rectangle2=Rectangle(-4,4)
# circle2=Circle(-4)
# triangle2=Triangle(4,4,4,"Hello error")
# triangle1.compute_perimeter_triangle()
# circle2.compute_surface_circle()
# rectangle2.compute_surface_rectangle()
