import gzip
import os
import json
import pickle
import re
from ast import literal_eval
import glob

DIR = 'raw_data'
DATA_PATH = 'dataset.pickle.gz'


if __name__ == "__main__":
    data = []
    feat = ['description', 'popular_shelves', 'similar_books', 'book_id']
    for fpath in glob.glob(os.path.join(DIR, '*')):
        category = re.findall(r'books_(\w+)\.json', fpath)[0]
        with gzip.open(f'{fpath}', 'rb') as f:
            for line in f:
                entry = json.loads(line)
                for k in set(entry.keys()) - set(feat):
                    del entry[k]
                entry['description'] = re.sub(
                    r'<.*>', '', entry['description'])
                entry['category'] = category
                data.append(entry)
                if data and len(data) % 30000 == 0:
                    break
        print(category, len(data))
    pickle.dump(data, gzip.open(DATA_PATH, 'wb'))
