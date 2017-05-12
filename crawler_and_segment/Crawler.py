
#-*-coding:utf-8-*-
from Download import dl
from urllib.parse import urlencode
print("import successfully")
import json, threading
def getUrl(page, key):
    url = 'https://s.taobao.com/api?'
    feed_dict = {
        # '_ksTS': '1492391041115_231',
        # 'm': 'customized',
        # 'ajax': 'true',
        'q': key,
        's': page,
        'rn': 'fcb53e078ddd635d41aa048a8a671207',
        'ie': 'utf8',
        'bcoffset': 0
    }
    return url + urlencode(feed_dict)

def getDetail(page, key, output):
    data = json.loads(dl.GetHtml(getUrl(page, key)))
    if data and data['API.CustomizedApi']['itemlist']['auctions']:
        for item in data['API.CustomizedApi']['itemlist']['auctions']:
            product_url = 'http:' + item['detail_url']
            product_title = item['raw_title']
            output.write(product_title+'\n')

def get(key):
    out_put_file = open(key + '.txt', 'w', encoding='utf8')
    print("lkasjdlgkjl")
    for page in range(0, 10000, 10):

        th = threading.Thread(target=getDetail, args=(page, key, out_put_file))
        th.start()

get(input('input keyword: '))
    
