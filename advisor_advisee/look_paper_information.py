
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
paper_file=open('refined_data/paper_information.txt','w',encoding='utf-8')
# institution_file=open('refined_data/author_institution.txt','w',encoding='utf-8')
author_file=open('refined_data/author_paper.txt','w',encoding='utf-8')
# author_institution=defaultdict(list)
author_paper=defaultdict(list)
count=0
with open('d:/dataset/paperbyfield/computer_science_papers.txt',encoding='utf-8') as file:
    for line in file:
        if count%100000==0:
            print(count)
        count+=1
        content=eval(line.strip())
        if content.__contains__('year') and content.__contains__('authors') and content['year']>=1970:
            is_used=False
            for aut in content['authors']:
                if aut['name'] in all_authors:
                    is_used=True
                    author_paper[aut['name']].append(count)
                    # if aut.__contains__('org'):
                    #     author_institution[aut['name']].append(aut['org'])
            if is_used:
                is_useful=False
                paper_info={}
                paper_info['year']=content['year']
                if content.__contains__('keywords'):
                    is_useful=True
                    paper_info['keywords']=content['keywords']
                if content.__contains__('fos'):
                    paper_info['fos']=content['fos']
                paper_file.write(str(count)+'\t'+str(paper_info)+'\n')
                paper_info.clear()

for aut,paper in author_paper.items():
    author_file.write(aut+'\t'+str(paper)+'\n')
author_file.close()

# for aut,inst in author_institution.items():
#     institution_file.write(aut+'\t'+str(inst)+'\n')
# institution_file.close()