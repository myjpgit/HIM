

import csv
'''useless'''
aut_year={}

out_file=open('data/false_advisor_advisee.csv','w',encoding='utf-8',newline='')
writer=csv.writer(out_file)

authors=set()
with open('data/true_advisor_advisee.csv',encoding='utf-8') as file:
    reader=csv.reader(file)
    for line in reader:
        authors.add(line[0])
        authors.add(line[1])

with open('temp_data/false_advisor_paper_number.txt',encoding='utf-8') as file:
    for line in file:
        line=line.strip().split('\t')
        paper=eval(line[2])
        year=list(paper.keys())[0]
        if year>=2000 and year<=2010:
           if line[0] in authors or line[1] in authors:
               writer.writerow([line[1],line[0],year])

out_file.close()