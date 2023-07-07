import requests
import zipfile
import os
import shutil
import tiktoken
import numpy as np

dataset_url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
dataset_path = os.path.dirname(__file__)
zip_path = os.path.join(dataset_path, 'cornell_movie_dialogs_corpus.zip')

split_ratio = 0.8

# Download the dataset
response = requests.get(dataset_url)
if response.status_code == 200:
    with open(zip_path, 'wb') as f:
        f.write(response.content)
else:
    print('Failed to download the dataset.')
    exit()

# Extract the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)

# Read the dialogues from the dataset
dialogues = []
with open(os.path.join(dataset_path, 'cornell movie-dialogs corpus/movie_lines.txt'), 'r', encoding='iso-8859-1') as f:
    for line in f:
        
        line_parts = line.strip().split(" +++$+++ ")
        if len(line_parts) == 5:
            dialogues.append(line_parts[-1])

dialogues = '\n'.join(dialogues)
split_index = int(len(dialogues) * split_ratio)
train_data = dialogues[:split_index]
val_data = dialogues[split_index:]

# gpt2 bpe
enconder = tiktoken.get_encoding('gpt2')
train_ids = np.array(enconder.encode_ordinary(train_data), dtype=np.uint16)
val_ids = np.array(enconder.encode_ordinary(val_data), dtype=np.uint16)

# Export to bin files
train_ids.tofile(os.path.join(dataset_path, 'train.bin'))
val_ids.tofile(os.path.join(dataset_path, 'val.bin'))

# Clean up the downloaded dataset and extracted files
os.remove(os.path.join(dataset_path, 'cornell_movie_dialogs_corpus.zip'))
shutil.rmtree(os.path.join(dataset_path, 'cornell movie-dialogs corpus'))
shutil.rmtree(os.path.join(dataset_path, '__MACOSX'))
