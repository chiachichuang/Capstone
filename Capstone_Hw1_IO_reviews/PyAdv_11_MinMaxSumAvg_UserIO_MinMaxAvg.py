"""
Assignment 11

Write a program to do the following:

1)	Prompts the user to enter his/her name
2)	Welcomes the user to the program
3)	Prompts the user to enter four floating point numbers
4)	Prints the largest and smallest of the four numbers
5)	Prints the sum of four numbers
6)	Prints the average of four numbers
"""
input_nums=[]
def getMinNMax():
    print(input_nums)
    return min(input_nums), max(input_nums)

def getAvg():
    return getSum()/len(input_nums)

def getSum():
    return sum(input_nums)

def main():
    userName = input("Please enter your name : ")
    print("Welcome to assignment 11, ", userName)
    inputs = input("Please enter four floating point numbers : ")
    global input_nums
    input_nums = [float(x) for x in inputs.split(" ")]
    #print(input_nums)
    minv, maxv = getMinNMax()
    print("largest and smallest of the four numbers : %.02f %.02f " %(maxv,minv))
    print("the sum of four numbers : %.02f" %getSum())
    print("the average of four numbers : %.02f" %getAvg())

if __name__ == "__main__":
    main()