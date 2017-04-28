from bs4 import BeautifulSoup as BS
import magic
import json
import os
import time
from pprint import pprint
from glob import glob
import io
import re
import unicodedata as ud
from numpy import *
import numpy as np

def merge_duplicates():
    maxcandidateid = 0
    files_ = glob('Candidate Profile Data/*')
    file_data = {}
    count = 0
    total = 0

    for j,i in enumerate(files_):
        with io.open(i, 'r', encoding='us-ascii') as f:
            data = f.read().lower()
            data = data.replace('\\n','') #remove \n
            data = re.sub(r'\\u[0-9a-f]{,4}','',data) # remove \uxxxx
            file_data[j] = json.loads(data)
            total += len(file_data[j])
    for i in range(len(files_)-1):
        list_i = {}
        for j in range(i+1,len(files_)):
            #print files_[i],files_[j]
            #print i,j
            one = file_data[i]
            two = file_data[j]
            list_j = {}
            for k in range(len(one)):
                for l in range(len(two)):
                    if one[k]['skills'] == two[l]['skills']:
                        #print "True"
                        list_i[k] = True
                        list_j[k] = True
                        count += 1

                        file_data[j][l]['candidateid'] = one[k]['candidateid']
                        for z in range(l+1,len(two)):
                            file_data[j][z]['candidateid'] = unicode(int(two[z]['candidateid'])-1)
                        #print one[k]['skills']
                        #print two[l]['skills']
                    maxcandidateid = int(two[-1]['candidateid'])
    """
            for k in list_j.keys():
                del file_data[j][k]
        for k in list_i.keys():
            del file_data[i][k]
    """
    for j,i in enumerate(files_):
        with io.open(i, 'w', encoding='utf8') as f:
            f.write(unicode(json.dumps(file_data[j])))
            f.truncate()
    return maxcandidateid
