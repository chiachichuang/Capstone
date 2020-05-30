"""
Assignment 13 Functions

Write a program which will do the following:

1)	Ask the user to enter five test scores
2)	Accept grades which are greater than 0 and less than or equal to 100
3)	Display the letter grade for each score and average test score.

Check with the sample run.

Your program MUST have the following functions:

getScore
This function should prompt the user to enter one score and return the score after it validates the score. Note that this function should be called 5 times from main.

calcAverage
This function should accept 5 scores as arguments and return the average of the scores.

determineGrade
This function should accept a test score as an argument and return the letter grade for
the score based on the following grading scale:
90-100 A, 80-89 B, 70-79 C, 60-69 D, below 60 F. Use if … elif … else in this function.

Main
This function should call the other 3 functions.
"""

score_record = []

def getScore():
    global score_record
    ipt = input("Please enter 5 scores between 0 and 100:")
    ipt_list = ipt.strip().split(" ")
    validated=True
    if len(ipt_list) != 5:
        print("You entered " + str(len(ipt_list)) + " score(s).")
        validated=False
        #getScore()
    else:
        for x in ipt_list:
            if not x.isdigit() or 0 > int(x) or int(x) > 100:
                print(x + " is not valid score")
                validated=False
                #getScore()
    if validated:
        score_record = [int(x) for x in ipt_list]
    else:
        tryagain=input("would you like to enter the scores? Y/N\n")
        if tryagain.lower()=="y":
            getScore()
        else:
            score_record=[]

def calcAverage():
    scoresavg = round(sum(score_record)/len(score_record))
    print("Average of 5 scores is : " + str(scoresavg))

def determineGrade():
    print("Score:Grade")
    print("=====================")
    for scr in score_record:
        if scr>=90:
            print(str(scr) + " : A")
        elif 90>scr>=80:
            print(str(scr) + " : B")
        elif 80>scr>=70:
            print(str(scr) + " : C")
        elif 70>scr>=60:
            print(str(scr) + " : D")
        else:
            print(str(scr) + " : F")
    print("=====================")

def main():
    getScore()
    if len(score_record) == 5:
        determineGrade()
        calcAverage()

if __name__=="__main__":
    main()