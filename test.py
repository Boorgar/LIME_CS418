#%%
import datasets
from tqdm import tqdm

agnews = datasets.load_dataset("ag_news", cache_dir="./datasets")
# g20news = datasets.load_dataset("SetFit/20_newsgroups", cache_dir="./datasets")
# yelp_reviews = datasets.load_dataset("yelp_review_full", cache_dir="./datasets")
# dbpedia = datasets.load_dataset("dbpedia_14", cache_dir="./datasets")

#%%
print(agnews)

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
import json
train_data = ['train']
test_data = ['test']

# Create training data
texts = train_data['text']
labels = train_data['label']

classes = train_data.unique('label')
classes.sort()
classes = [str(c) for c in classes]
data = [{'text': text, 'label': label} for text, label in zip(texts, labels)]

out_data = {"classes": classes, "data": data}
with open('_train.json', "w", encoding="utf-8") as wf:
    json.dump(out_data, wf, indent=4, ensure_ascii=False)


# Create test data
texts = train_data['text']
labels = train_data['label']

classes = train_data.unique('label')
classes.sort()
classes = [str(c) for c in classes]
data = [{'text': text, 'label': label} for text, label in zip(texts, labels)]

out_data = {"classes": classes, "data": data}
with open('agnews_test.json', "w", encoding="utf-8") as wf:
    json.dump(out_data, wf, indent=4, ensure_ascii=False)

#%%
import json
test_output = []
with open('output.txt') as file:
    line = file.readline().split()
with open('agnews_test.json') as file:
    data = json.load(file)

test_y = []
for i in data['data']:
    test_y.append(i['label'])

counter = 0
total = 0
for i in range(len(line)):
    total += 1
    if int(line[i]) == test_y[i]:
        counter += 1
    else:
        print(line[i], test_y[i])
print(counter/total)
# %%
