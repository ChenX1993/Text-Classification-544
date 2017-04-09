import math
import os
import sys

inputfile = sys.argv[1]
scalefile = sys.argv[2]
lower = 0
upper = 1

max_index = -sys.maxint - 1
num_nonzero = 0

new_num_nonzero = 0
dic = {}
y_max = -sys.maxint - 1
y_min = sys.maxint
y_scaling = 0
feature_max = []
feature_min = []
filecontent = []
line0 = ""
line1 = ""
line2 = ""
line3 = ""
with open(inputfile,'r') as infile:
	lines = infile.readlines()
	line0 = lines[0]
	line1 = lines[1]
	line2 = lines[2]
	line3 = lines[3]
	for i in range(4,len(lines)):
		line = lines[i].strip()
		items = line.split(" ")
		y_max = max(y_max, int(items[0]))
		y_min = min(y_min, int(items[0]))
		items = items[1:]
		filecontent.append(items)
	infile.close()

for content in filecontent:
	for one_item in content:
		index = int(one_item.split(":")[0])
		max_index = max(index, max_index)
		num_nonzero += 1
for i in range(0,max_index+1):
	feature_max.append(-sys.maxint - 1)
	feature_min.append(sys.maxint)
next_index = 1
for content in filecontent:
	for one_item in content:
		pair = one_item.split(":")
		index = int(pair[0])
		value = float(pair[1])
		for i in range(next_index, index):
			feature_max[i] = max(feature_max[i], 0)
			feature_min[i] = min(feature_min[i], 0)
		feature_max[index] = max(feature_max[index], value)
		feature_min[index] = min(feature_min[index], value)
		next_index += 1
for i in range(next_index, max_index+1):
	feature_max[i] = max(feature_max[i], 0)
	feature_min[i] = min(feature_min[i], 0)

outputcontent = ""
def output(index, value):
	global new_num_nonzero
	if(feature_max[index] == feature_min[index]):
		return value
	if(value == feature_min[index]):
		value = lower
	elif(value == feature_max[index]):
		value = upper
	else:
		value = lower + (upper-lower)*(value-feature_min[index])/(feature_max[index]-feature_min[index])
		value = round(value,6)
	if(value!=0):
		new_num_nonzero += 1
	return value
next_index = 1
with open(inputfile, 'r') as infile:
	lines = infile.readlines()
	for k in range(4,len(lines)):
		line = lines[k].strip()
		items = line.split(" ")
		print items
		label = items[0]
		print label
		outputcontent += label+" "
		for i in range(1, len(items)):
			pair = items[i].split(":")
			index = int(pair[0])
			value = float(pair[1])
			# for i in range(next_index, index):
			# 	result1 = output(i, 0)
			# 	if (result1!=0):
			# 		outputcontent += str(result1) +":"
			outputcontent += str(index) + ":"
			result2 = output(index,value)
			if (result2!=0.0):
				outputcontent += str(result2) + " "
			next_index += 1
			# for i in range(next_index,max_index+1):
			# 	result3 = output(i, 0)
			# 	if (result3!=0):
			# 		outputcontent += str(result3) + " "
		outputcontent +='\n'
	#if (new_num_nonzero > num_nonzero):
	infile.close()
with open(scalefile, 'w') as outfile:
	outfile.write(line0)
	outfile.write(line1)
	outfile.write(line2)
	outfile.write(line3)
	outfile.write(outputcontent)
	outfile.close()




