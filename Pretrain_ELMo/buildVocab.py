from __future__ import print_function
from os import listdir
from os.path import isfile, join
from collections import Counter

train_path = "./data/PMC_Case_Rep/train/"
train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]

if __name__ == '__main__':
	vocab = {}

	for file in train_files:
		with open(train_path+file, 'r', encoding="ISO-8859-1") as f:
			lines = f.readlines()
			lines = [line.strip() for line in lines]
			for line in lines:
				words = line.split()
				for word in words:
					if word in vocab:
						vocab[word] = vocab[word] + 1
					else:
						vocab[word] = 1

	print(vocab)

	with open('./data/newVocab.txt', 'w') as f:
		for item in vocab:
			f.write("%s\n" % item)