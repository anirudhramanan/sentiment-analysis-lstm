from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

dataset_folder_path = 'aclImdb_v1.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('aclImdb_v1.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Dataset') as pbar:
        urlretrieve(
            'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
            'aclImdb_v1.tar.gz',
            pbar.hook)

if not isdir(dataset_folder_path):
    with tarfile.open('aclImdb_v1.tar.gz') as tar:
        tar.extractall()
        tar.close()