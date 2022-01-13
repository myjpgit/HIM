

import csv
from collections import defaultdict

all_authors=set()

with open('refined_data/refined_true_aa.csv',encoding='utf-8') as file:
    reader=csv.reader(file)
    for line in reader:
        all_authors.add(line[0])
        all_authors.add(line[1])

with open('refined_data/cut_false_aa_by_diff_year.csv',encoding='utf-8') as file:
    reader=csv.reader(file)
    for line in reader:
        all_authors.add(line[0])
        all_authors.add(line[1])
print(len(all_authors))
institution_file=open('refined_data/author_institution.txt','w',encoding='utf-8')
author_institution=defaultdict(list)
count=0
with open('d:/dataset/paperbyfield/computer_science_papers.txt',encoding='utf-8') as file:
    for line in file:
        if count%100000==0:
            print(count)
        count+=1
        content=eval(line.strip())
        if content.__contains__('year') and content.__contains__('authors'):
            for aut in content['authors']:
                if aut['name'] in all_authors:
                    if aut.__contains__('org') and content['year']>=2000 and content['year']<=2010:
                        author_institution[aut['name']].append(aut['org'])

for aut,inst in author_institution.items():
    institution_file.write(aut+'\t'+str(inst)+'\n')
institution_file.close()