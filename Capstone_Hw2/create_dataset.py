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
#from PIL import image

def get_args():
    import sys
    num_samples = int(sys.argv[1])
    output_file = sys.argv[2]
    img_file_name = sys.argv[3]
    return num_samples,output_file,img_file_name

#######################################
# create_dataset creates the dataset
# based on user parameters
#######################################
def create_dataset(num_samples,output_file):
    import random
    random.seed(9999) # makes the dataset repeatable
    # most common male names
    males = ["liam","noah","william","james","oliver"]
    # most common female names
    females = ["emma","olivia","sophia","camila","hannah"]
    names = [] # combine all names
    names.extend(males)
    names.extend(females)

    # file to write your dataset
    fopen = open(output_file,"w")

    # percent salary increase every year
    salary_increase = 2

    # dictionary to store names and counts
    d_name_count = {}

    # Initialize the dictionary with 0 counts
    for name in names:
        d_name_count[name] = 0
        
    # loop for user specified line count
    while (num_samples > 0):
        # last four digits of SSN
        ssn = random.randint(1234,9999)
        # pick one of the names from collection
        name = random.choice(names)
        # keep count for names
        d_name_count[name] += 1
        # Birthday month
        bday_month = random.randint(1,12)
        # age of that person
        age = random.randint(25,75)
        # salary: start at 50k and given raise every year
        salary = 50000 + ((50000 * salary_increase/100.0) * (age - 25))
        # make a list and convert to strings
        concat = [ssn,name,bday_month,age,salary]
        concat = [str(item) for item in concat]
        # write record to file
        write_str = ",".join(concat)
        fopen.write(write_str + "\n")
        # decrement sample count
        num_samples -= 1
    fopen.close() # close file after writing
    return d_name_count # return dictionary to main function

#######################################
# make_bar_chart creates the bar chart
# for the variables in the dictionary.
#######################################

#import Image
#import matplotlib.pyplot as plt

def make_bar_plot(d_name_count, img_file_name):
    import matplotlib.pyplot as plt
    # get the list of names and counts
    names = list(d_name_count.keys())
    values = list(d_name_count.values())
    # plotting horizontal bar chart
    plt.barh(names,values)
    # place count numbers next to bars
    for name,value in zip(names,values):
        plt.text(value,name,str(value))
    # save the plot
    #plt.savefig("dataset_profile.jpg",dpi=600,format="jpg"
    plt.savefig(img_file_name+".jpg", dpi=600, format="jpg")
    # show the plot
    plt.show()
    plt.close()

#######################################
# main: calls other functions
#######################################   
def main():
    num_samples,output_file, img_file_name = get_args() # getting args
    #num_samples, output_file = get_args()  # getting args
    d_name_count = create_dataset(num_samples,output_file) # create dataset
    make_bar_plot(d_name_count, img_file_name) # make bar plot

#######################################
# call the main function
#######################################  
if __name__ == "__main__":
    main()
    

