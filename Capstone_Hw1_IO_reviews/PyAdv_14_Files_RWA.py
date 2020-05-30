"""
# Assignment 14
=========== Assignment instructions
The main() and show_instructions() functions are already written.
Your job is to write the following functions, which are documented in the file:

Create the file “mylist.txt” which has two entries:
line 1 can be aaa, line 2 can be bbb.
Now the functions from below interact with the file “mylist.txt.”

1.	add_item(item)
2.	remove_item(item) - This one is a little tricky. You will need to open the file twice, once for reading and once for writing. You may also find the splitlines() method of a string object useful.
3.	delete_list()
4.	print_list()
"""

def add_item(item):
    pass
    """Appends item (after stripping leading and trailing
            whitespace) to list.txt followed by newline character

            Keyword arguments:
            item -- the item to append"""
    with open('./mylist.txt', 'a') as myfile:
        myfile.write(item)

def remove_item(item):
    pass
    """Removes first instance of item from list.txt
    If item is not found in list.txt, alerts user.

    Keyword arguments:
    item -- the item to remove"""
    contents=""
    needEdit=True
    with open('mylist.txt', 'r') as myfile:
        contents = myfile.read()
        if item not in contents:
            needEdit = False
            print("Can not find " + item + " in the list")
        else:
            contents = contents.replace(item, "", 1)

    if needEdit:
        with open('mylist.txt', 'w') as myfile:
            myfile.write(contents)


def delete_list():
    """Deletes the entire contents of the list by opening
    list.txt for writing."""
    with open('mylist.txt', 'w') as myfile:
        myfile.write("")

def print_list():
    """Prints list"""
    with open('mylist.txt', 'r') as myfile:
        contents = myfile.read()
        if len(contents)==0:
            print("List is empty")
        else:
            print("List is " + contents)


# Do not modify the following functions
def show_instructions():
    """Prints instructions"""
    print("""OPTIONS:
    	P
        	-- Print List
    	+abc
        	-- Add 'abc' to list
    	-abc
        	-- Remove 'abc' from list
    	--all
        	-- Delete entire list
    	Q
        	-- Quit\n""")


def main():
    show_instructions()

    while True:
        choice = input('>> ')

        if choice.lower() == 'q':
            print('Goodbye!')
            break
        elif choice.lower() == 'p':
            print_list()
        elif choice.lower() == '--all':
            delete_list()
        elif len(choice) and choice[0] == '+':
            add_item(choice[1:])
        elif len(choice) and choice[0] == '-':
            remove_item(choice[1:])
        else:
            print("I didn't understand.")
            show_instructions()

if __name__ == '__main__':
    main()