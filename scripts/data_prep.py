import os
import csv

def get_files(search_path):
     for (dirpath, _, filenames) in os.walk(search_path):
         for filename in filenames:
             yield filename

print('creating training csv set')

header = ["file_name", "target (0: negative - 1: positive)"]

training_set = open('./aclImdb/train/train.csv', 'w')
test_set = open('./aclImdb/test/test.csv', 'w')

with training_set:
    positive_review_files = get_files('./aclImdb/train/pos')
    negative_review_files = get_files('./aclImdb/train/neg')
    writer = csv.writer(training_set)
    writer.writerow(header)
    
    for pos_file in positive_review_files:
        writer.writerow([pos_file, 1])
        data_limit = data_limit - 1

    for neg_file in negative_review_files:
        writer.writerow([neg_file, 0])
        data_limit = data_limit - 1


# with test_set:
#     positive_review_files = get_files('./aclImdb/test/pos')
#     negative_review_files = get_files('./aclImdb/test/neg')
#     writer = csv.writer(test_set)
#     writer.writerow(["file_name"])

#     for pos_file in positive_review_files:
#         if data_limit == 0:
#             data_limit = 3000
#             break;
#         else:
#             writer.writerow(['pos/' + pos_file])
#             data_limit = data_limit - 1


#     for neg_file in negative_review_files:
#         if data_limit == 0:
#             data_limit = 3000
#             break;
#         else:
#             writer.writerow(['neg/' + neg_file])
#             data_limit = data_limit - 1

print('training.csv created')
