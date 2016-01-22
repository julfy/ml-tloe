import json
import glob
import re
from urlparse import urlparse
from pprint import pprint

def check_match (s1, s2):
    s1l = re.findall(r"[\w']+", s1.lower())
    s2l = re.findall(r"[\w']+", s2.lower())
    return len(set(s1l) & set(s2l))

def process(jsn):
    i = 0
    for res in jsn['urls']:
        if i < 10:
            label = 1
        else:
            label = 0
        common_content = check_match (jsn['keyword'], res['content'])
        common_url = check_match (jsn['keyword'],urlparse(res['url']).path+urlparse(res['url']).hostname)
        url_len = len(re.sub(r"www\.","", urlparse(res['url']).hostname))
        print [res['ahrefs_rank'], res['domain_rating'],common_url,common_content,url_len,label]
        i += 1
        

with open('/home/julfy/work/results/6004ba3055b15f6b11f79bade79dcc31') as json_data:
    d = json.load(json_data)
    json_data.close()
    process(d)

    
# filelist = glob.glob("/home/julfy/work/results/*")

# for filename in filelist:
#     j += 1
