import csv
import pandas as pd
from disease_similarity import semantic_similarity
import os
count = 0
d_uid = []
d_ms = []
d_set = []
d_mh = []
name2index = dict()
files_home = '../files'
file = os.path.join(files_home, 'd2019.bin')
f = open(file,'r',encoding = "ISO-8859-1")
line = f.readline()

# Parsing all the disease descriptors (whose MN begin with c) from d2019.bin
while line :

    while line and (line[:10] != '*NEWRECORD'):
        line = f.readline()

    while line and (line[:4] != 'MH ='):
        line = f.readline()
    line = line[:-1]
    try:
        d_name = line.split(' = ')[1]
    except Exception as e:
        break

    while line and (line[:4] != 'MN ='):
        line = f.readline()
    adds = set()
    while line and (line[:4] == 'MN =') :
        line = line[:-1]
        if line.split(' = ')[1][0] == 'C' :
            adds.add(line.split(' = ')[1])
        line = f.readline()

    while line and (line[:4] != 'MS ='):
        line = f.readline()
    line = line[:-1]
    ms = line.split(' = ')[1]

    while line and (line[:4] != 'UI ='):
        line = f.readline()
    line = line[:-1]
    uid = line.split(' = ')[1]

    if len(adds)!=0:
        # adds: tree address, ms: scope note, uid: Unique ID, d_name: disease name
        d_set.append(adds)
        d_ms.append(ms)
        d_uid.append(uid)
        d_mh.append(d_name)
        name2index[d_name] = count
        count = count + 1
        if count % 1000 == 0 :
            print(count)

print(count)
f.close()

all_concept = set()
m_concepts = pd.read_csv(os.path.join(files_home,'mesh_concepts_in_gdas.csv')) # derived from DisGeNet, only contains MeSH concepts
for i in range(0,len(m_concepts)):
    all_concept.add(m_concepts['CID'][i])

count = 0
c_uid = []
c_set = []
c_mh = []

f = open(os.path.join(files_home,'c2018.bin'),'r',encoding = "ISO-8859-1")
line = f.readline()

# Parsing all the disease concepts (whose UI in all_concepts) from c2019.bin
while line :

    while line and (line[:10] != '*NEWRECORD') :
        line = f.readline()

    while line and (line[:4] != 'NM ='):
        line = f.readline()
    line = line[:-1]
    try:
        c_name = line.split(' = ')[1]
    except Exception as e:
        break

    while line and (line[:4] != 'HM ='):
        line = f.readline()
    heads = set()
    while line and (line[:4] == 'HM =') :
        line = line[:-1]
        if len(line.split(' = *')) != 2 :
            line = f.readline()
            continue
        heads.add(line.split(' = *')[1])
        line = f.readline()

    while line and (line[:4] != 'UI ='):
        line = f.readline()
    line = line[:-1]
    uid = line.split(' = ')[1]

    # the Concepts is no given tree number, only provide name of linked Descriptors
    # So we assign a tree number according to the record dictionary with key name
    if (uid in all_concept) and (len(heads)!=0) :
        adds = set()
        for head in heads:
            if head in name2index.keys():
                tmp_index = name2index[head]
            else :
                continue
            head_tree = d_set[tmp_index]
            for ht in head_tree:
                adds.add(ht+'.c'+str(count))  # add 'c' and use a global ID to ensure the unique tree ID
        if len(adds) == 0 :
            print('nooononooo')
            continue
        c_set.append(adds)
        c_uid.append(uid)
        c_mh.append(c_name)
        count = count + 1
        if count % 1000 == 0 :
            print(count)

df_dict = pd.DataFrame(columns=('Unique ID','MeSH Heading','Tree Number','Scope Note'))
filename_dict = os.path.join(files_home,'mesh_brief_dict.csv')
df_dict.to_csv(filename_dict)

csvfile = open(filename_dict, 'a',encoding = "ISO-8859-1")
writer = csv.writer(csvfile)
for i in range(0,len(d_uid)):
    tn_string = ','.join(d_set[i])
    writer.writerow([i,d_uid[i],d_mh[i],tn_string,d_ms[i]])



k = i+1
for i in range(0,len(c_uid)):
    tn_string = ','.join(c_set[i])
    writer.writerow([k+i,c_uid[i],c_mh[i],tn_string,''])
csvfile.close()

df_similarity = pd.DataFrame(columns=('UID1','UID2','score'))
filename_sml = os.path.join(files_home,'mesh_similarity.csv')
df_similarity.to_csv(filename_sml)

threshold = 0.05
csvfile = open(filename_sml, 'a')
writer = csv.writer(csvfile)
for i in range(0,len(d_set)-1):
    for j in range(i+1,len(d_set)):
        score = semantic_similarity(d_set[i],d_set[j])
        if score > threshold :  ###
            writer.writerow(['dd',d_uid[i],d_uid[j],'%.6f'%score])
for i in range(0,len(d_set)):
    for j in range(0,len(c_set)):
        score = semantic_similarity(d_set[i],c_set[j])
        if score > threshold :  ###
            writer.writerow(['dc',d_uid[i],c_uid[j],'%.6f'%score])
for i in range(0,len(c_set)-1):
    for j in range(i+1,len(c_set)):
        score = semantic_similarity(c_set[i],c_set[j])
        if score > threshold :  ###
            writer.writerow(['cc',c_uid[i],c_uid[j],'%.6f'%score])
csvfile.close()