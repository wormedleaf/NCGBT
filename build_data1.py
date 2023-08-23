cc_list = []
f = open('datas/mozilla_my_cc_map.txt', 'r')
lines = f.readlines()
for line in lines:
    cc_list.append(line.strip())
f.close()

doc_cc_map = {}
for doc_id in range(len(cc_list)):
    cc_ids = cc_list[doc_id].rstrip('\n').split(' ')
    for cc_id in cc_ids:
        doc_cc_map.setdefault(doc_id, []).append(cc_id)

doc_cc_map = sorted(doc_cc_map.items())

cc_doc = []
for temp in doc_cc_map:
    for cc_id in temp[1]:
        cc_doc.append(str(temp[0]) + '\t' + str(cc_id))
cc_doc_str = '\n'.join(cc_doc)

f = open('datas/mozilla_my_cc_doc.txt', 'w')
f.write(cc_doc_str)
f.close()