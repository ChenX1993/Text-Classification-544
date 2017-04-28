# def a(path, n):
# 	lines = list()
# 	with open(path, 'r') as f:
# 		lines = f.readlines()
# 	print len(lines[2])
# 	with open('1' + path, 'w') as f:
# 		count = 0
# 		for line in lines:
# 			if count >= n:
# 				break
# 			if len(line) < 50:
# 				continue
# 			f.write(line)
# 			count+=1

#a('C000020_1.txt', 750)
with open('C000013_dev.txt') as f:
	print (len(f.readlines()))