"""
Assignment 15

Create a new Python script to take 2 command line arguments. The name of the script will be:
“Print_alpha_nums.py”

The first argument could be the alphabet “A” or “N”
A = alphabets
N = numbers

The second argument could be “F” or “B”
F = forward
B = backward

Running the script should produce the following outputs:
1)	python.exe Print_alpha_nums.py A F
A B C D E F G H I J

2)	python.exe Print_alpha_nums.py A B
J I H G F E D C B A

3)	python.exe Print_alpha_nums.py N F
1 2 3 4 5 6 7 8 9 10

4)	python.exe Print_alpha_nums.py N B
10 9 8 7 6 5 4 3 2 1

Assume that the user will supply a valid first input of A or N and a valid second input of F or B.
"""

def printAlpha(isForward):
    if isForward:
        print(" ".join([chr(x) for x in range(ord('A'), ord('K'))]))
    else:
        print(" ".join([chr(x) for x in range(ord('J'), ord('A')-1, -1)]))

def printNum(isForward):
    if isForward:
        print(" ".join([str(x) for x in range(1,11)]))
    else:
        print(" ".join([str(x) for x in range(10, 0, -1)]))

def show_instructions():
    """Prints instructions"""
    print("""OPTIONS:
    	A F
        	-- Print A B C D E F G H I J
    	A B
        	-- Print J I H G F E D C B A
    	N F
        	-- Print 1 2 3 4 5 6 7 8 9 10
    	N B
        	-- Print 10 9 8 7 6 5 4 3 2 1
    	Q
        	-- Quit\n""")

def main():
    show_instructions()
    while True:
        choice = input('>> ')

        if choice.lower() == 'a f':
            printAlpha(True)
        elif choice.lower() == 'a b':
            printAlpha(False)
        elif choice.lower() == 'n f':
            printNum(True)
        elif choice.lower() == 'n b':
            printNum(False)
        elif choice.lower() == 'q':
            print('Goodbye!')
            break
        else:
            print("I didn't understand.")
            show_instructions()

if __name__=="__main__":
    main()