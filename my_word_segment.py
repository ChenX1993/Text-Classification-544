# -- coding: utf-8 --
import jieba
import jieba.posseg as posseg
import os

input_path = 'SogouC/ClassFile/C000020_short_test'
output_path = 'SogouC/Segment/C000020_short_test'

tag_list = ['t','q','p','u','e','y','o','w','m']

def segment(content):
    result = ""
    results = ""
    words = posseg.cut(content)
    count = 0
    for word in words:
        #print "org: "+tmp
        if len(word.word)>1 and len(word.flag)>0 and word.flag[0] not in tag_list and  word.word[0]>=u'\u4e00' and word.word[0]<=u'\u9fa5':
            result = result + " " + word.word
            count += 1
        if count%100 == 0:
            #print re
            #re = re.replace("\n"," ")
            results = results + "\n" + result
            result = ""
            count += 1
    result = result.replace("\n"," ").replace("\r"," ")   
    if len(results)>1 and len(result)>40:
        results = results + "\n" + result
    elif len(result)>40:
        results = result
    results = results + "\n"
    results = results.replace("\r\n","\n").replace("\n\n","\n")

    return results

def getTrainData(input_path,output_path):
    #fp = open(outfile,"a") 
    for filename in os.listdir(input_path):
        outFileName = output_path+ "/" + filename[:-4] + "_cut.txt"
        print outFileName
        with open(outFileName, 'w') as fo:
            with open(input_path + "/" + filename, 'r') as fi:
                content = fi.read()
                content = segment(content)
                if len(content) > 30:
                    fo.write(content.encode("utf-8"))

getTrainData(input_path,output_path)
