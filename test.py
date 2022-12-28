#%%
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

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

texts = train_data['text']
labels = train_data['label']

classes = train_data.unique('label')
classes.sort()
classes = [str(c) for c in classes]
data = [{'text': text, 'label': label} for text, label in zip(texts, labels)]

import json

out_data = {"classes": classes, "data": data}
with open('test.json', "w") as wf:
    json.dump(out_data, wf, indent=4, ensure_ascii=False)

#%%
with open('test.json') as rf:
    data = json.load(rf)

print(data['classes'])
print(data['data'][0])

#%%
import create_pseudo_labels

classify = create_pseudo_labels.

correct = total = 0
samples = []
for i, sample in enumerate(tqdm(data["data"])):
    text = sample["text"]
    prediction, probs = classify(model, tokenizer, classes, verbalizer, text)
    # predicted_class = classes[prediction]
    if prediction.item() == sample["label"]:
        correct += 1
    total += 1
    prob = probs[prediction]
    out_sample = {
        "label": sample["label"],
        "prediction": prediction.item(),
        "confidence": probs,
        "text": sample["text"],
    }
    samples.append(out_sample)
print(f"Stats for {args.model_type}, {args.out_file}: {correct / total}")
