import simplejson as json
import glob
import re
import csv
from urlparse import urlparse
from pprint import pprint

def check_match (s1, s2):
    s1l = re.findall(r"[\w']+", s1.lower())
    s2l = re.findall(r"[\w']+", s2.lower())
    return len(set(s1l) & set(s2l))

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key,value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def process(jsn):
    i = 0
    data = []
    labels = []
    for res in jsn['urls']:
        if i < 10:
            label = 1
        else:
            label = 0
        common_content = check_match (jsn['keyword'], res['content'])
        common_url = check_match (jsn['keyword'],urlparse(res['url']).path+urlparse(res['url']).hostname)
        url_len = len(re.sub(r"www\.","", urlparse(res['url']).hostname))
        input = [res['ahrefs_rank'], res['domain_rating'],common_url,common_content,url_len]
        data.append(input)
        labels.append([label])
        i += 1
    return data,labels

def transform_data(dir, outd, outl):
    filelist = glob.glob(dir)
    for filename in filelist:
        print "Reading ", filename
        with open(filename) as json_data:
            d = json.load(json_data)
            json_data.close()
            data,labels = process(d)
            with open(outd, 'ab') as f:
                writer = csv.writer(f)
                writer.writerows(data)
            with open(outl, 'ab') as f:
                writer = csv.writer(f)
                writer.writerows(labels)
    print "Processed %d files", len(filelist)

transform_data ("/home/bogdan/Downloads/results/*", 'data', 'labels')
