import os
import csv
import matplotlib.pyplot as plt

def get_files(search_path):
     for (dirpath, _, filenames) in os.walk(search_path):
         for filename in filenames:
             yield filename

positive_review_files = get_files('./aclImdb/train/pos')
negative_review_files = get_files('./aclImdb/train/neg')

review_wordcount = []

for file_name in positive_review_files:
	file = open('./aclImdb/train/pos/'+file_name, "r")
	review = file.read()
	for split in review.split():
		review_wordcount.append(split)

for file_name in negative_review_files:
	file = open('./aclImdb/train/neg/'+file_name, "r")
	review = file.read()
	for split in review.split():
		review_wordcount.append(split)

max(set(review_wordcount), key=review_wordcount.count)

# plt.cla()
# plt.title('Reviews vs Words')
# plt.xlabel('Words')
# plt.ylabel('Number of reviews')
# plt.gca().set_yscale('log', nonposy='clip')
# plt.grid(True)
# count, bins, ignored = plt.hist(review_wordcount, 100, histtype='stepfilled')
# plt.show()
