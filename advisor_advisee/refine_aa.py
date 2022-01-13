
import csv
from collections import defaultdict


def get_author_year():
    author_year={}
    with  open('temp_data/all_authors_paper_number.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip().split('\t')
            years=list(eval(line[1]).keys())
            years=sorted(years)
            author_year[line[0]]=years[0]
    return author_year

def refine_true_aa():
    advisor_advisee=defaultdict(list)
    with open('data/true_advisor_advisee.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            advisor_advisee[line[1]].append(line[0])
    with open('refined_data/refined_true_aa.csv','w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file)
        for advisor,advisee in advisor_advisee.items():
            if len(advisee)>=3:
                for adv in advisee:
                    writer.writerow([adv,advisor])

def refine_false_aa():
    authors = set()
    with open('refined_data/refined_true_aa.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            authors.add(line[0])
            authors.add(line[1])
    out_file=open('refined_data/refined_false_aa.csv','w',encoding='utf-8',newline='')
    writer=csv.writer(out_file)
    with open('data/false_advisor_advisee.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            if line[0] in authors or line[1] in authors:
                writer.writerow(line)

def cut_false_aa_by_diff_year():
    authors = set()
    with open('refined_data/refined_false_aa.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            authors.add(line[1])
    out_file=open('refined_data/cut_false_aa_by_diff_year.csv','w',encoding='utf-8',newline='')
    writer=csv.writer(out_file)
    author_year=get_author_year()
    with open('refined_data/refined_false_aa.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            # year=int(line[-1])
            if abs(author_year[line[0]]-author_year[line[1]])<=5:
                writer.writerow(line)
    out_file.close()



if __name__=='__main__':
    # refine_true_aa()
    # refine_false_aa()
    cut_false_aa_by_diff_year()
