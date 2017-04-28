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
from numpy.matlib import *
import merge_ids_duplicates as mid

def get_model():
    """
    This code will clean the data and assign unique candidate ids to each candidates
    """

    files = glob('Candidate Profile Data/*')
    data = dict()
    skillset = dict()
    skillsetjob = dict()
    candidate = 0

    for file_ in files:
        with io.open(file_, 'r+', encoding='utf-8') as json_file:
            json_data = json_file.read().lower()

            json_data = ud.normalize('NFKC',json_data)#.encode('ascii','ignore')
            filename = "".join(file_.split('.')[:-1]).split('/')[1]
            # remove all non-ascii chars
            json_data = json_data.replace('\\n','') #remove \n
            json_data = re.sub(r'\\u[0-9a-f]{,4}','',json_data) # remove \uxxxx
            json_file.seek(0)
            #json_data = json_data.encode('utf8')

            #print json_data
            json_data = json.loads(json_data)

            j = 0
            m = 0
            for i in range(len(json_data)):
                skills_data = json_data[i]['skills']
                skills = skills_data.split(',')
                json_data[i]['candidateid'] = unicode(candidate)
                # Regex for (excluding curved brackets and commas): (10+ years), (5 years), (Less than 1 year)
                skill_experience_years = re.findall(r'((\d\d(?=\+ years\)))|(\d(?= (years|year)\))))',skills_data)
                for l in skill_experience_years:
                    j += 1
                # skill_respective = re.findall() (\((?=\d\d\+ years\)))|(\((?=\d years\)))|(\((?=Less than 1 year\)))|(\((?=1 year\)))

                #Regex for skills of the above years of experience by removing the experience part and replacing it with
                #a unique token ('token') based on which individual skills would be extracted, respectively:

                skills = re.sub(r'\s\(((\d\d(\+ years))|(\d (years|year))|(less than 1 year))\)', 'token', skills_data)
                skills = re.sub(r'(token\,\s)','token',skills)
                skills = skills.split('token')
                skills = skills[:][:-1] # last is empty
                for index,skl in enumerate(skills):
                    m += 1
                    #print l
                    if skl in skillset.keys():
                        skillset[skl] += 1
                        skillsetjob[filename] = 1
                    else:
                        skillset[skl] = 1
                        skillsetjob[filename] = 1
                candidate += 1
            #print j
            #print m
            if j!=m:
                print "Error\n"
            else:
                #print "OK\n"
                pass
            #print filename
            #break
            json_file.write(unicode(json.dumps(json_data)))
            json_file.truncate()

    #pruned_skillset = {}
    for i in skillset.keys():
        if skillset[i] > 15:
            #print i,skillset[i]
            pass
        else:
            del skillset[i]

    #print "Keys:",len(skillset.keys())
    #y = zeros((,len(files_)))
    maxcandidateid = mid.merge_duplicates()

    # limit skillsets
    pruned_skillset = {}
    pruned_skillsetjob = {}
    for file_ in files:
        with io.open(file_, 'r+', encoding='utf-8') as json_file:
            json_data = json_file.read().lower()

            filename = "".join(file_.split('.')[:-1]).split('/')[1]
            json_data = json.loads(json_data)

            j = 0
            m = 0
            for i in range(len(json_data)):
                skills_data = json_data[i]['skills']
                skills = skills_data.split(',')
                candidateid = json_data[i]['candidateid']
                #Regex for skills of the above years of experience by removing the experience part and replacing it with
                #a unique token ('token') based on which individual skills would be extracted, respectively:

                skills = re.sub(r'\s\(((\d\d(\+ years))|(\d (years|year))|(less than 1 year))\)', 'token', skills_data)
                skills = re.sub(r'(token\,\s)','token',skills)
                skills = skills.split('token')
                skills = skills[:][:-1] # last is empty
                for index,skl in enumerate(skills):
                    m += 1
                    #print l
                    if skl in skillset.keys():
                        pruned_skillset[candidateid,skl] = 1
                        pruned_skillsetjob[candidateid,filename] = 1
            #print filename
            #break

    #print len(pruned_skillset)
    #print len(pruned_skillsetjob)
    #print pruned_skillset.keys()
    #print pruned_skillsetjob.keys()
    user_job = repmat(np.nan,maxcandidateid+1,len(skillsetjob))
    user_skill = repmat(0,maxcandidateid+1,len(skillset))

    skillset_numeric = {}
    skillsetjob_numeric = {}
    for j,i in enumerate(skillset.keys()):
        skillset_numeric[i] = j
    for j,i in enumerate(skillsetjob.keys()):
        skillsetjob_numeric[i] = j

    for i in pruned_skillsetjob:
        jobid = int(skillsetjob_numeric[i[1]])
        user_job[int(i[0]),jobid] = 1
    for i in pruned_skillset:
        skillid = int(skillset_numeric[i[1]])
        user_skill[int(i[0]),skillid] = 1
    #print user_job.shape, user_skill.shape
    return user_job, user_skill, skillset_numeric, skillsetjob_numeric
