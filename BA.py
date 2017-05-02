from random import randint
import random
import urllib, json
import numpy
i=0 
res=[]
k=0
def jsonDefault(object):
    return object.__dict__
url = "http://fitnessiot.mybluemix.net/getdata"
response = urllib.urlopen(url)
data = json.loads(response.read())
print len(data)
for i in range(len(data)) :
	#print data[i]['d']['heartBeatting'],data[i]['d']['temperature'],data[i]['d']['tension'] 
	if (60< data[i]['d']['heartBeatting'] <100) and (36< data[i]['d']['temperature'] <37.5) and (9<data[i]['d']['tension'] <14):
		res.append({'heartBeatting' : data[i]['d']['heartBeatting'],'temprerature' : data[i]['d']['temperature'],'tension':data[i]['d']['tension'],'sport':1})
	else :
		res.append({'heartBeatting' : data[i]['d']['heartBeatting'],'temprerature' : data[i]['d']['temperature'],'tension':data[i]['d']['tension'],'sport':0})
with open('data.json', 'w') as outfile:
    json.dump(res, outfile)