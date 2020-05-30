"""
Assignment 12 Functions

import random
 def main():
    pass #replace this with your code

main()

1.	Write the main() function so that it:
1.	Creates a sequence with three elements: "Rock", "Paper", and "Scissors".
2.	Makes a random choice for the computer and stores it in a variable.
3.	Prompts the user with 1 for Rock, 2 for Paper, 3 for Scissors:
4.	Prints out the computer's choice and then the user's choice.
Sample Run:
1 for Rock, 2 for Paper, 3 for Scissors
Computer: Paper
User: Rock
"""
import random

rps_choices=["Rock", "Paper","Scissors"]

def getRandomRPS() -> int:
    return random.randrange(0,2,1)

def main():
    computer_picked = getRandomRPS()
    user_picked = input("Please enter 1 for Rock, 2 for Paper, 3 for Scissors :")
    print("Computer : " + rps_choices[computer_picked])
    print("User : " + rps_choices[int(user_picked)-1])

if __name__ == "__main__":
    main()