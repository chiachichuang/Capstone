####################
# PyAdv_S02_Args.py
####################
#
# Program to process script arguments
#

import sys


def show_numbers():
    print("From show_numbers: 12345678910 ...")


# this is the main function
def main():
    print("Number of arguments ...", len(sys.argv))
    for a, b in enumerate(sys.argv):
        print(a, b)
    # call the function show numbers
    show_numbers()


if __name__ == "__main__":
    main()
