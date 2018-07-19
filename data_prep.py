import os
import csv

def get_files(search_path):
     for (dirpath, _, filenames) in os.walk(search_path):
         for filename in filenames:
             yield filename

print('creating training csv set')

header = ["file_name", "target (0: negative - 1: positive)"]

training_set = open('./dataset/train/train.csv', 'w')
test_set = open('./dataset/test/test.csv', 'w')

with training_set:
    positive_review_files = get_files('./dataset/train/pos')
    negative_review_files = get_files('./dataset/train/neg')
    writer = csv.writer(training_set)
    writer.writerow(header)

    for pos_file in positive_review_files:
    	writer.writerow([pos_file, 1])

    for neg_file in negative_review_files:
    	writer.writerow([neg_file, 0])

with test_set:
    positive_review_files = get_files('./dataset/test/pos')
    negative_review_files = get_files('./dataset/test/neg')
    writer = csv.writer(test_set)
    writer.writerow(["file_name"])

    for pos_file in positive_review_files:
        writer.writerow(['pos/' + pos_file])

    for neg_file in negative_review_files:
        writer.writerow(['neg/' + neg_file])

print('training.csv created')
