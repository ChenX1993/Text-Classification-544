
#-*-coding:utf-8-*-
import json, threading
from Download import dl
from urllib.parse import urlencode

def getUrl(page, keyword):
    start_url = 'https://s.taobao.com/api?'
    add_dict = {
        '_ksTS': '1492391041115_231',
        'm': 'customized',
        'ajax': 'true',
        'q': keyword,
        'rn': 'fcb53e078ddd635d41aa048a8a671207',
        'ie': 'utf8',
        's': page,
        'bcoffset': 0
    }
    return start_url + urlencode(add_dict)

def getDetail(page, keyword, output):
    data = json.loads(dl.GetHtml(getUrl(page, keyword)))
    if data and data['API.CustomizedApi']['itemlist']['auctions']:
        for item in data['API.CustomizedApi']['itemlist']['auctions']:
            detail_url = 'http:' + item['detail_url']
            detail_title = item['raw_title']
            output.write(detail_title+'\n')

def get(keyword):
    out_put_file = open(keyword + '.txt', 'w', encoding='utf8')
    for page in range(0, 10000, 10):
        th = threading.Thread(target=getDetail, args=(page, keyword, out_put_file))
        th.start()

if __name__ == '__main__':
    get(input('input keyword: '))
