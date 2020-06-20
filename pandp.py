import re
"""
1.	Approximately how many words are in pride-and-prejudice.txt?
2.	What is the most common vowel to show up in pride-and-prejudice.txt?
"""

def getWordnVowelCount():
    with open("pride-and-prejudice.txt", "r", encoding='utf-8') as myfile:
        contents = myfile.readlines()

    word_count = 0
    #word_count2 = 0
    vowel_count = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0}
    for line in contents:
        if "from http" not in line.lower():
            line2 = line.replace("\n", "").replace("\b", " ").replace("-", " ").replace("—", " ").replace('"','').strip()
            n2, n1 = len(re.findall(r'[\w]+', line2)), len(line2.split(" "))
            n3 = len(re.findall(r"[\w+]'[\w+]", line2))
            #if n1 != n2-n3 and n2!=0 and testc < 20:
            #word_count2 += n1
            word_count += n2-n3

            for vc in vowel_count:
                vc2 = vc.upper()
                vowel_count[vc] += len(re.findall(r"[" + vc + vc2 + "]", line))

    return {"word_count": word_count, "vowel_count":vowel_count}

def main():
    result = getWordnVowelCount()
    print("Approximately " + str(result["word_count"]) + " words in pride-and-prejudice.txt")
    mostcnt, mstcntvowel=0, 'a'
    print("Each Vowel counts are ::")
    for vowelc, vowelcnt in result["vowel_count"].items():
        if vowelcnt > mostcnt:
            mostcnt = vowelcnt
            mstcntvowel = vowelc
        print("Vowel (" + vowelc + ") " + str(vowelcnt) + " times")
    print("Vowel " + mstcntvowel + " has the most counts, " + str(mostcnt) + ", among all vowels.")

main()