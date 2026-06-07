import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B/")
ids = []
with open("/data/sets/tinyshakespeare/input.txt") as f:
    for line in f:
        tokens = tokenizer.encode(line.strip())
        ids.extend(tokens)

arr = np.array(ids, dtype=np.int32)
arr.tofile("./data/tokens.bin")
print(f"共 {len(ids)} 个 token，文件大小 {len(ids)*4/1e6:.2f} MB")
