import json

data = {"hello": "world"}

with open("/home/pritok_llm/data/sft_train.json", "w") as json_file:
    json.dump(data, json_file)



