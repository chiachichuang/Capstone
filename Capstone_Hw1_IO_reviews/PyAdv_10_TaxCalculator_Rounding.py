"""
Assignment 10

A simple tax calculator:

Write a Python program that:
1)	Ask the use for his/her name
2)	Displays a welcome message using users name
3)	Prompts for the price of an item
4)	Computes the sales tax to be charged
5)	Computes the total cost
6)	Prints the sales tax to be charged and total cost
7)	Thanks the user for using the program

Notes:
Create a constant variable which has the sales tax rate: 8.5%

Valid inputs:
Assume the user will enter a non-negative number.

Rounding:
Round all outputs to 2 digits after the decimal point.
"""
import io
import sys

def addTax(orgprice, taxrate):
    addedTax = orgprice * taxrate
    return addedTax, orgprice + addedTax

def main():
    userName = input("Please enter your name:")
    print("Welcome to assignment 10, ", userName)
    itemPrice = float(input("Please enter the price of your item : "))
    saleTax, totalCost = addTax(itemPrice, 0.085)
    print("Sale tax of your item is $%.2f " %saleTax)
    print("Total cost of your item is $%.2f " %totalCost)

if __name__ == "__main__":
        main()