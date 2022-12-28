#%%
import datasets
from datamodule import PseudoDataset
from torch.utils.data import DataLoader

agnews = datasets.load_dataset("ag_news", cache_dir="./datasets")
g20news = datasets.load_dataset("SetFit/20_newsgroups", cache_dir="./datasets")
yelp_reviews = datasets.load_dataset("yelp_review_full", cache_dir="./datasets")
dbpedia = datasets.load_dataset("dbpedia_14", cache_dir="./datasets")

#%%
print(agnews['clas'])

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
# json:
# {
#   "classes": ["World", "Sports", "Business", "Sci/Tech"],
#  "data": [
#   {
#    "label": 3,
#   "text": "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."
#  },
# ...
# ] 
# }

#%%
train_data = agnews['train']
test_data = agnews['test']

train_data