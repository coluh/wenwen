import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(
    "./Qwen2.5-0.5B/", local_files_only=True
)
print(tokenizer.bos_token_id, tokenizer.eos_token_id)

with open("/data/sets/TinyStories/TinyStoriesV2-GPT4-train.txt") as f:
    raw_text = f.read()

stories = [s.strip() for s in raw_text.split("<|endoftext|>")]
stories = [s for s in stories if s]

stories = stories[:1000000]
story_lens = sorted([(len(s), s, i) for (i, s) in enumerate(stories)])
for l, s, i in story_lens[:3]:
    print(f"[{i}] {l} : {s}")
exit(0)

all_lens = []
total_tokens = 0
with open("./data/tokens.bin", "wb") as f:
    for story in tqdm(stories, desc="Tokenizing"):
        tokens = tokenizer.encode(story, add_special_tokens=False)
        tokens.append(151643)
        all_lens.append(len(tokens))
        total_tokens += len(tokens)
        np.array(tokens, dtype=np.int32).tofile(f)

print(f"共 {len(stories)} 个序列，最短 {min(all_lens)}，最长 {max(all_lens)}，共 {total_tokens} 个 token")
