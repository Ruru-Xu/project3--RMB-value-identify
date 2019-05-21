#-*- coding:utf-8 -*-
import csv
from itertools import islice

label = ['0.1','0.2','0.5','1','2','5','10','50','100']
csv_file = csv.reader(open('submit-2019052102.csv','r'))
# print(csv_file)

f_submit = open('submit2-2019052102.csv', 'w')
f_submit.write('name,label\n')

for stu in islice(csv_file, 1, None):
    # print(stu)
    # print(type(stu))
    #print(max(stu[1:10]))
    # max_value = max(stu[1:10])
    max_value_index = stu.index(max(stu[1:10]))
    max_value_index_label = label[max_value_index-1]
    print(stu[0] + ',' + max_value_index_label)
    f_submit.write('%s,%s\n' % (stu[0], max_value_index_label))
print("Done!!!!")
f_submit.close()

