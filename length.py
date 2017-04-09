# -- coding: utf-8 --
import numpy 
import array
import re

tfidfPath = 'model/test.svm'

with open(tfidfPath, 'r') as f:
	lines = f.readlines()
	print len(lines)