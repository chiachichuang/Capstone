########################################################
# Python script to create synthetic controlled dataset
# Sanjay Agarwal v1 05252020
########################################################

# next line to support error in new pyinstaller
# next line should be removed in future
import pkg_resources.py2_warn

#######################################
# get_args gets arguments and parsers
# them to correct variable names.
#######################################

import matplotlib


# from PIL import image

def get_args():
    import sys
    num_samples = int(sys.argv[1])
    output_file = sys.argv[2]
    img_file_name = sys.argv[3]
    return num_samples, output_file, img_file_name


#######################################
# create_dataset creates the dataset
# based on user parameters
#######################################
def create_dataset(num_samples, output_file):
    import random
    random.seed(9999)  # makes the dataset repeatable
    # most common male names
    males = ["liam", "noah", "william", "james", "oliver"]
    # most common female names
    females = ["emma", "olivia", "sophia", "camila", "hannah"]
    names = []  # combine all names
    names.extend(males)
    names.extend(females)

    # file to write your dataset
    fopen = open(output_file, "w")

    # percent salary increase every year
    salary_increase = 2

    # dictionary to store names and counts
    d_name_count = {}
    d_age_salary = {"ages":[], "salaries":[]}

    # Initialize the dictionary with 0 counts
    for name in names:
        d_name_count[name] = 0
    gender_list = random.choices(["male", "female"], weights=[45, 55], k=num_samples)
    """
            a new column “Education” to the dataset: 
            This should be a number in the range [1,4]. 
            1 = High School, 2 = Bachelors, 3 = Masters, 4 = Doctorate. 
            The distribution should be 
            40% for high-school, 40% for Bachelors, 15% for Masters, and 5% for Doctorate
    """
    education_list = random.choices([1, 2, 3, 4], [40, 40, 15, 5], k=num_samples)

    # loop for user specified line count
    while num_samples > 0:
        # last four digits of SSN
        ssn = random.randint(1234, 9999)
        # pick one of the names from collection

        if gender_list[num_samples-1] == "male":
            name = random.choice(males)
        else:
            name = random.choice(females)

        # keep count for names
        d_name_count[name] += 1
        # Birthday month
        bday_month = random.randint(1, 12)
        # age of that person
        age = random.randint(25, 75)
        # education of that person
        education = education_list[num_samples-1]
        # salary
        years = age-25
        salary = 50000
        while years > 0:
            salary *= (1+(1+education)/100)
            salary = round(salary, 0)
            years -= 1
        d_age_salary["ages"].append(age)
        d_age_salary["salaries"].append(salary)
        # make a list and convert to strings
        concat = [ssn, name, bday_month, age, education, salary]
        concat = [str(item) for item in concat]
        # write record to file
        write_str = ",".join(concat)
        fopen.write(write_str + "\n")
        # decrement sample count
        num_samples -= 1
    fopen.close()  # close file after writing
    return d_name_count, d_age_salary  # return dictionaries to main function


#######################################
# make_bar_chart creates the bar chart
# for the variables in the dictionary.
#######################################

# import Image
# import matplotlib.pyplot as plt

def make_scatter_plot(d_age_salary):
    import matplotlib.pyplot as plt
    # get the list of ages and salaries
    ages = list(d_age_salary["ages"])
    salaries = list(d_age_salary["salaries"])
    plt.scatter(ages, salaries)
    plt.xlabel('age')
    plt.ylabel('salary')
    # save the plot
    plt.savefig("age_salary.jpg", dpi=600, format="jpg")

def make_bar_plot(d_name_count, img_file_name):
    import matplotlib.pyplot as plt
    # get the list of names and counts
    names = list(d_name_count.keys())
    values = list(d_name_count.values())
    # plotting horizontal bar chart
    plt.barh(names, values)
    # place count numbers next to bars
    for name, value in zip(names, values):
        plt.text(value, name, str(value))
    # save the plot
    plt.savefig(img_file_name + ".jpg", dpi=600, format="jpg")
    # show the plot
    #plt.show()
    #plt.close()


#######################################
# main: calls other functions
#######################################
def main():
    num_samples, output_file, img_file_name = get_args()  # getting args
    # num_samples, output_file = get_args()  # getting args
    d_name_count,d_age_salary = create_dataset(num_samples, output_file)  # create dataset
    make_bar_plot(d_name_count, img_file_name)  # make bar plot
    make_scatter_plot(d_age_salary) #make scatter plot


#######################################
# call the main function
#######################################
if __name__ == "__main__":
    main()


