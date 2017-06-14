import numpy as np
import json 
import time

if False:
    data = []
    data.append({
        'index' : 0,
        'array' : [0,0,0]
        })

    data.append({
        'index' : 1,
        'array' : [1,1,1]
        })

    with open('./data/test.json', 'w') as f:
        json.dump(data, f)

with open('./data/test.json') as f:
    data = json.load(f)

for i in range(10):
    if not any(d['index'] == i for d in data):
        time.sleep(1)
        print(i)
        data.append({
            'index' : i,
            'array' : [i,] * 3
            })

        with open('./data/test.json', w) as f:
            json.dump(data, f)



