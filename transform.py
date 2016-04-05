import simplejson as json
import glob
import re
import csv
import os
from urlparse import urlparse
from pprint import pprint

rex = re.compile("\n|\s|,|\.|\?|!|:|\t|-|_|;|\(|\)|\{|\}|\[|\]|/")
def mysplit(str):
    return filter(None, rex.split (str))

def check_match (keywords, s2):
    s1l = set(keywords)
    s2l = set(s2)
    res = (float(len(s1l & s2l)) / len(s1l)) if len(s1l) > 0 else 0
    return res

def process(jsn): #still no anchors
    i = 0
    data = []
    for res in jsn['urls']:
        urlp = urlparse(res['url'])
        s_key = mysplit(jsn['keyword'].lower())
        s_content = mysplit(res['content'].lower())
        s_url = mysplit (urlp.path.lower())
        s_title = mysplit(res['title'].lower())
        s_h1 = mysplit(str(res['h1']).lower())
        s_h2 = mysplit(str(res['h2']).lower())
        s_domain = mysplit(urlp.hostname.lower())

        m_content = check_match (s_key, s_content)
        # e_content = int(m_content)
        m_url = check_match (s_key, s_url)
        # e_url = int(m_url)
        m_title = check_match(s_key, s_title)
        m_h1 = check_match(s_key, s_h1)
        m_h2 = check_match(s_key, s_h2)
        m_domain = check_match(s_key, s_domain)
        m_anchor =0
        for a in res['anchors']:
            if check_match(s_key,mysplit(a.get('anchor',"").lower())) > 0:
                m_anchor += a.get('backlinks',0)

        url_len = len(urlp.hostname) - (4 if urlp.hostname[:4] == 'www.' else 0)
        title_len = len(s_title)
        h1_len = len(s_h1)
        h2_len = len(s_h2)
        content_len = len(s_content)

        content_occurence = sum([1 if word in jsn['keyword'] else 0 for word in s_content])

        metrics = res['metrics']
        social = [0,0,0,0,0,0,0,0,0] if res['social'] == [] else res['social']
        # label = (i if i <= 100 else 100) / 100.0
        label = 1 if i < 10 else 0


        inp = ([label] +
               social +








               [res['ahrefs_rank'], #mb take average if -1
                res['domain_rating'], #mb take average if -1
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
                m_content,
                m_url,
                m_title,
                m_h1,
                m_h2,
                m_domain,
                m_anchor,
                url_len,
                title_len,
                h1_len,
                h2_len,
                content_len,
                content_occurence])
        data.append(inp)
        i += 1
    return data

def transform_data(dir, outd, num=0):
    try:
        os.mkdir(outd, 0755)
    except:
        print "ERROR: \"", outd, "\" already exists"
        return
    filelist = glob.glob(dir)
    i = 0
    failed = 0
    outs = 0
    limit = 100000 #rows
    cur = 0
    print "Transforming", dir
    for filename in filelist:
        if num > 0 and i >= num:
            break
        with open(filename) as json_data:
            d = ""
            try:
                s = json_data.read()
                d = json.loads(s)
                json_data.close()
            except:
                print "WARNING: ", filename, "malformed, ignoring"
                failed = failed + 1
                continue
            data = process(d)
            if len(data) == 0:
                continue
            if cur > limit:
                outs = outs + 1
                cur = 0
            cur = cur + len(data)
            with open(outd+'/'+str(outs), 'ab') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        i = i + 1
    print "Transformed", i, "files (", failed, "failed ) to", outs + 1, "files"
