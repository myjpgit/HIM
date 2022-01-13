

import csv
'''useless'''
def cut_false_aa():
    author_year={}
    with  open('temp_data/all_authors_paper_number.txt',encoding='utf-8') as file:
        for line in file:
            line=line.strip().split('\t')
            years=list(eval(line[1]).keys())
            years=sorted(years)
            author_year[line[0]]=years[0]

    out_file=open('data/cut_false_aa.csv','w',encoding='utf-8',newline='')
    writer=csv.writer(out_file)

    with open('data/false_advisor_advisee.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            year=int(line[-1])
            if year-author_year[line[0]]>5:
                writer.writerow(line)

    out_file.close()

def cut_true_aa():
    out_file=open('data/cut_true_aa.csv','w',encoding='utf-8',newline='')
    writer=csv.writer(out_file)
    with open('data/true_advisor_advisee.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            year=int(line[-1])
            if not year<=2003:
                writer.writerow(line)
    out_file.close()

def cut_false_aa_by_true():
    authors = set()
    with open('data/cut_true_aa.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            authors.add(line[0])
            authors.add(line[1])
    out_file=open('data/cut_false_aa_by_true.csv','w',encoding='utf-8',newline='')
    writer=csv.writer(out_file)
    with open('data/cut_false_aa.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            if line[0] in authors or line[1] in authors:
                writer.writerow(line)


def cut_paper_information(all_papers):
    out_file=open('temp_data/cut_paper_information.txt','w',encoding='utf-8')
    with open('temp_data/paper_information.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            ids=int(con[0])
            if ids in all_papers:
                out_file.write(line)
    out_file.close()

def cut_institution(all_authors):
    out_file=open('temp_data/cut_author_institution.txt','w',encoding='utf-8')
    with open('temp_data/author_institution.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            if con[0] in all_authors:
                out_file.write(line)
    out_file.close()

def cut_author_paper(all_authors):
    out_file=open('temp_data/cut_author_paper.txt','w',encoding='utf-8')
    all_papers=set()
    with open('temp_data/author_paper.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            if con[0] in all_authors:
                papers=set(eval(con[1]))
                all_papers|=papers
                out_file.write(line)
    print(len(all_papers))
    out_file.close()
    cut_paper_information(all_papers)

def cut_inst_paper():
    all_authors=set()
    with open('data/cut_true_aa.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            all_authors.add(line[0])
            all_authors.add(line[1])
    with open('data/cut_false_aa_by_true.csv',encoding='utf-8') as file:
        reader=csv.reader(file)
        for line in reader:
            all_authors.add(line[0])
            all_authors.add(line[1])
    print(len(all_authors))
    # cut_institution(all_authors)
    cut_author_paper(all_authors)


def cut_paper_information_by_year():
    out_file=open('temp_data/cut_paper_information_by_year.txt','w',encoding='utf-8')
    with open('temp_data/cut_paper_information.txt',encoding='utf-8') as file:
        for line in file:
            con=line.strip().split('\t')
            info=eval(con[1])
            if info['year']<=2010 and 1970<=info['year']:
                out_file.write(line)
    out_file.close()

if __name__=="__main__":
    # cut_false_aa()
    # cut_true_aa()
    # cut_false_aa_by_true()
    # cut_inst_paper()
    cut_paper_information_by_year()