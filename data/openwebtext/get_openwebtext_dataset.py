# Adapted from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset


def encode_text(example, encoder):
    ids = encoder.encode_ordinary(example["text"]) + [encoder.eot_token]
    return {"ids": ids, "len": len(ids)}


def main():
    # Load and split the dataset
    dataset = load_dataset("openwebtext")
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    # Tokenize the dataset
    enc = tiktoken.get_encoding("gpt2")
    tokenized = split_dataset.map(
        lambda example: encode_text(example, enc),
        remove_columns=["text"],
        desc="Tokenizing the splits",
    )

    # Concatenate tokenized data and save to files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16 if enc.max_token_value < 2**16 else np.uint32
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format(
                "numpy"
            )
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


if __name__ == "__main__":
    main()
