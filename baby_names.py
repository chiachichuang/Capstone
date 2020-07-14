import sys

def readData(file_name_list):
    decade_dir, init_dir = {}, {}
    for filename in file_name_list:
        curr_rank = 1
        curr_target_line = "<tr ><td>1</td>"
        year_input = filename.replace("s.html", "").replace("names", "").strip()
        decade_dir[year_input] = []
        tp_list = []
        with open(filename, "r") as myfile:
            curr_line = myfile.readline()
            while curr_line and curr_rank<201:
                if curr_line.startswith(curr_target_line):
                    curr_rank += 1
                    curr_target_line = "<tr ><td>" + str(curr_rank) + "</td>"
                    for idx in range(3):
                        curr_line = myfile.readline()
                    curr_line_list = curr_line.replace("</td></tr>\n", "").replace(" ","").split("</td><td>")
                    curr_line_list[0] = curr_line_list[0].replace("<td>", "")
                    if curr_line_list[0] not in init_dir:
                        init_dir[curr_line_list[0]] = {"boy":{year_input: [curr_line_list[1], curr_rank]},"girl":{}}
                        #init_dir[curr_line_list[0]]["boy"] = {year_input: [curr_line_list[1], curr_rank]}
                    else:
                        init_dir[curr_line_list[0]]["boy"][year_input] = [curr_line_list[1], curr_rank]
                    if curr_line_list[2] not in init_dir:
                        init_dir[curr_line_list[2]] = {"girl": {year_input: [curr_line_list[3], curr_rank]}, "boy": {}}
                        #init_dir[curr_line_list[2]]["girl"] = {year_input: [curr_line_list[3], curr_rank]}
                    else:
                        init_dir[curr_line_list[2]]["girl"][year_input] = [curr_line_list[3], curr_rank]
                    tp_list.append(curr_line_list)
                curr_line = myfile.readline()
        decade_dir[year_input] = tp_list
        # print("**************************************")
        # print(decade_dir)
        # print("**************************************")
    return decade_dir, init_dir

def showMenu():
    print("""Menu:
    	(d) Search by decade 
    	(i) Search by initials
    	(q) quit\n""")

def main():
    is_valid_input = False
    while not is_valid_input:
        file_names = input("Options names1990s.html, names2000s.html , names2010s.html\nEnter comma separated file names: ")
        file_names_list = file_names.split(",")
        err_msg = ""
        for file_name in file_names_list:
            if file_name not in ["names1990s.html", "names2000s.html", "names2010s.html"]:
                err_msg += file_name + " "
        if err_msg == "":
            is_valid_input = True
        else:
            print("Invalid input : " + err_msg)
            if input("quit? Y/N").lower() !="n":
                exit()

    names_by_decade, names_by_init = readData(file_names_list)
    for key_year in names_by_decade:
        print("Found 200 names for each gender in file names"+key_year+"s.html")

    input_choice = 'cc'
    while input_choice not in ['q','d','i']:
        showMenu()
        input_choice = input("Your choice:").lower().strip()
        if input_choice not in ['q','d','i']:
            print("Invalid choice")

        if input_choice == "q":
            exit()
        elif input_choice == "d":
            sub_choice = "1900"
            while sub_choice not in names_by_decade.keys():
                sub_choice = input("Enter decade(" + ", ".join(names_by_decade.keys()) +") >> ")
            curr_idx = 0
            while curr_idx < 200:
                print(" ".join(names_by_decade[sub_choice][curr_idx]))
                continue_decade = input("Press Enter for next names or q for Quit >> ")
                if continue_decade.lower() == "q":
                    exit()
                curr_idx += 1
            input_choice = 'cc'
        else:
            input_init = input("Enter a comma separated list of initials >> ")
            input_init_list = input_init.split(",")
            names_list = []
            for name in names_by_init.keys():
                name0 = name.lower()
                for init0 in input_init_list:
                    upb = min(len(name), len(init0))
                    if name0[:upb] == init0[:upb]:
                        for gender_input in ["boy","girl"]:
                            info = ""
                            if len(names_by_init[name][gender_input]) > 0:
                                if "1990" in names_by_init[name][gender_input]:
                                    info += "(" + gender_input + ") : rank " + str(names_by_init[name][gender_input]["1990"][1]) + " in 1990, "
                                if "2000" in names_by_init[name][gender_input]:
                                    info += "(" + gender_input + ") : rank " + str(names_by_init[name][gender_input]["2000"][1]) + " in 2000, "
                                if "2010" in names_by_init[name][gender_input]:
                                    info += "(" + gender_input + ") : rank " + str(names_by_init[name][gender_input]["2010"][1]) + " in 2010, "
                                info = info[:len(info)-2]
                                names_list.append([name, info])
            if len(names_list) == 0:
                print("Not found any name with your search initials : " + input_init)
            else:
                names_list.sort()
                for name_item in names_list:
                    print(name_item[0] + ": " + name_item[1])
            input_choice = 'cc'


main()