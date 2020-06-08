def get_args():
    import sys
    file_name = sys.argv[1]
    first_name_filter = sys.argv[2]
    first_name_filter = first_name_filter.lower()
    nn_or_edu = sys.argv[3]
    out_file = sys.argv[4]
    return file_name, first_name_filter, nn_or_edu, out_file

def extract_data(file_name, first_name_filter, nn_or_edu, output_file):
    # <hs> or <bac> or <mas> or <doc> to filter by education
    education_map={"hs": "1", "bac": "2", "mas": "3", "doc": "4"}
    checknn = True
    if nn_or_edu in education_map.keys():
        checknn = False
        edu_filter = nn_or_edu
    else:
        nn = nn_or_edu
    write_contents = []
    with open(file_name, "r") as myfile:
        contents = myfile.readlines()

    for temp_data_raw in contents:
        temp_data = temp_data_raw.split(",")
        NotSelected = True
        if first_name_filter == "all" or first_name_filter == temp_data[1]:
            NotSelected = False
            if checknn == False:
                # filter by education
                if education_map[edu_filter] != temp_data[4]:
                    NotSelected = True
            else:
                # filter by SSN nn
                num_occurs=0
                for numstr in temp_data[0]:
                    if numstr == nn:
                        num_occurs += 1
                if num_occurs != 2:
                    NotSelected = True

        if NotSelected == False:
            write_contents.append(temp_data_raw)

    with open(output_file, "w") as outfile:
        outfile.writelines(write_contents)


#######################################
# main: calls other functions
#######################################
def main():
    file_name, first_name_filter, nn_or_edu, out_file = get_args()
    extract_data(file_name, first_name_filter, nn_or_edu, out_file)



#######################################
# call the main function
#######################################
if __name__ == "__main__":
    main()