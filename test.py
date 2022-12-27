#%%
import datasets
from datamodule import PseudoDataset
from torch.utils.data import DataLoader

agnews = datasets.load_dataset("ag_news", cache_dir="./datasets")
g20news = datasets.load_dataset("SetFit/20_newsgroups", cache_dir="./datasets")
yelp_reviews = datasets.load_dataset("yelp_review_full", cache_dir="./datasets")
dbpedia = datasets.load_dataset("dbpedia_14", cache_dir="./datasets")

#%%
print("Sample data\n")

print("Train sample")
print(agnews['train'][0]['text'])
print(agnews['train'][0]['label'])

print()

print("Test sample")
print(agnews['test'][0]['text'])
print(agnews['test'][0]['label'])

#%%
batch_size = 32
cpu_count = 8
tokenizer = None
supervised = False
confidence_threshold = 0.9

data = agnews['train']

for sample in data:
    print(sample)
    break

# filtered = [
#     sample
#     for sample in data["data"]
#     if sample["confidence"][sample["prediction"]] >= confidence_threshold
# ]

# data["data"] = filtered
train_set = PseudoDataset(data, tokenizer, use_pseudo=not supervised)
train_dataloader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=cpu_count // 4,
)
