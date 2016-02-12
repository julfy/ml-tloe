import simplejson as json
import glob
import re
import csv
import os
from urlparse import urlparse
from pprint import pprint

def check_match (keywords, s2):
    s1l = set(re.findall(r"[\w']+", keywords.lower()))
    s2l = set(re.findall(r"[\w']+", s2.lower()))
    res = (float(len(s1l & s2l)) / len(s1l)) if len(s1l) > 0 else 0
    return res

def process(jsn):
    i = 0
    data = []
    for res in jsn['urls']:
        common_content = check_match (jsn['keyword'], res['content'])
        common_url = check_match (jsn['keyword'],urlparse(res['url']).path+urlparse(res['url']).hostname)
        url_len = len(re.sub(r"www\.","", urlparse(res['url']).hostname))
        metrics = res['metrics']
        social = [0,0,0,0,0,0,0,0,0] if res['social'] == [] else res['social']
        inp = social + [res['ahrefs_rank'],
                        res['domain_rating'],
                        metrics.get('backlinks',0),
                        metrics.get('refpages',0),
                        metrics.get('pages',0),
                        metrics.get('text',0),
                        metrics.get('image',0),
                        metrics.get('not_sitewide',0),
                        metrics.get('nofollow',0),
                        metrics.get('dofollow',0),
                        metrics.get('edu',0),
                        metrics.get('rss',0),
                        metrics.get('html_pages',0),
                        metrics.get('links_internal',0),
                        metrics.get('links_external',0),
                        metrics.get('refdomains',0),
                        metrics.get('refclass_c',0),
                        metrics.get('refips',0),
                        metrics.get('linked_root_domains',0),
                        common_url,
                        common_content,
                        url_len]
        data.append(inp)
        i += 1
    return data

def transform_data(dir, outd, num=0):
    os.mkdir(outd, 0755)
    filelist = glob.glob(dir)
    i = 0
    failed = 0
    for filename in filelist:
        if num > 0 and i >= num:
            break
        with open(filename) as json_data:
            d = ""
            try:
                s = json_data.read().decode('utf-8')
                d = json.loads(s)
                json_data.close()
            except:
                print "Error in : ", filename
                failed = failed + 1
                continue
            data = process(d)
            if len(data) == 0:
                continue
            with open(outd+'/'+str(i), 'ab') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        i = i + 1
    print "Transformed", i, "files (", failed, "failed )"
