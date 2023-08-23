
import collections


cc_list = []
f = open('datas/eclipse/corpus/eclipse_my_method_0.5_cc.txt', 'r')
lines = f.readlines()
for line in lines:
    cc_list.append(line.strip())
f.close()

doc_cc_map = {}
for i in range(len(cc_list)):
    cc_ids = cc_list[i].rstrip('\n').split(' ')
    for cc_id in cc_ids:
        doc_cc_map.setdefault(int(cc_id), []).append(i)

doc_cc_map= sorted(doc_cc_map.items())

cc_doc = []
for temp in doc_cc_map:
    for doc_id in temp[1]:
        cc_doc.append(str(temp[0]) + '\t' + str(doc_id))
cc_doc_str = '\n'.join(cc_doc)

f = open('datas/eclipse_my_cc_doc.txt', 'w')
f.write(cc_doc_str)
f.close()
