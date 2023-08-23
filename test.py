cc_list = []
 = open('datas/eclipse_my_method_0.5_cc_doc.txt', 'r')
lines = f.readlines()
for line in lines:
    cc_list.append(line.strip())
f.close()
doc_id = set()
for i in range(len(cc_list)):
    cc_ids = cc_list[i].rstrip('\n').split('\t')
    doc_id.add(int(cc_ids[1]))

print(doc_id)