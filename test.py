import os

newfile = ""
inpath = 'SogouC/ClassFile/C000020_short/9.txt'
# for filename in os.listdir(inpath):
# 	print filename
file_object = open(inpath)
all_the_text = file_object.read()
all_the_text = all_the_text.decode("gb2312").encode("utf-8")
print all_the_text
newfile += all_the_text
with open(inpath,'w') as f:
	f.write(newfile)
	f.close()