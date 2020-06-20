import re
from datetime import datetime
"""
What is the difference in years between
the publication of the dates of
the oldest and newest novels listed in
the best-novels.txt file?
"""
def get_oldest_n_newest_published_info(filename):
    year_published = {}
    with open(filename, "r") as myfile:
        contents = myfile.readlines()
    oldest_pub_year, newest_pub_year = datetime.now().year, 1
    for line in contents:
        result = re.search(r"\t[0-9]{4}\n$", line)
        if result:
            pubyear = int(result.group(0))
            if pubyear not in year_published:
                year_published[str(pubyear)]=[]
            year_published[str(pubyear)].append(line)
            oldest_pub_year = min(oldest_pub_year, pubyear)
            newest_pub_year = max(newest_pub_year, pubyear)
    return oldest_pub_year, newest_pub_year, year_published

def main():
    oldest_pub_year,newest_pub_year, pub_info = get_oldest_n_newest_published_info("best-novels.txt")
    print("=======  Oldest Novels  =======")
    for novel in pub_info[str(oldest_pub_year)]:
        print(novel)
    print("=======  Newest Novels  =======")
    for novel in pub_info[str(newest_pub_year)]:
        print(novel)

main()