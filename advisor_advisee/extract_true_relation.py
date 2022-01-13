
import csv
'''useless'''

years=[i for i in range(2000,2011)]

res_file=open('data/true_advisor_advisee.csv','w',encoding='utf-8',newline='')
writer=csv.writer(res_file)
path='e:/Wang Lei/advisor_statistic/data/computer_science/dataset/'
for year in years:
    with open(path+str(year)+'_ins.csv',encoding='utf-8') as file:
        file.readline()
        reader=csv.reader(file)
        for line in reader:
            if int(line[-1])==1:
                writer.writerow(line[:2]+[year])

res_file.close()