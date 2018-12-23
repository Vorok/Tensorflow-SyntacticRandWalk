import os
import util
import re
# note: multiprocessing is messed up in jupyter
from multiprocessing import Pool

if __name__ == '__main__':
	file = open("data/enwik8.txt", "r")
	doclist = [line for line in file]
	docstr = ''.join(doclist)
	sentences = re.split(r'[.!?]', docstr)
	sentences = [sentence for sentence in sentences if len(sentence) > 1]

	pool = Pool(processes=10)
	sentences_parsed = pool.map(util.clean_up, sentences)

	with open("data/enwik8_cleaned.txt", 'w') as f:
		for sentence in sentences_parsed:
			for word in sentence:
				f.write("%s " % word)
			f.write(".\n")