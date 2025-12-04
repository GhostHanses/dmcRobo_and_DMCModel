import json

import numpy as np
from sympy.stats.rv import probability


class MyMessage:
    def __init__(self, name, content, weight=0):
        self.name = name
        self.content = content
        self.weight = weight
    def to_string(self):
        # return ("{}: {}").format(self.name, self.content)
        msg_dict = dict[str, str]()
        msg_dict["name"] = self.name
        msg_dict["content"] = self.content
        msg_dict["weight"] = self.weight
        return str(msg_dict)

class ChatHistory:
    def __init__(self, size):
        self.size = size
        self.contents = dict[str, list[MyMessage]]()

    def add_content(self, group_id, content: MyMessage):
        if group_id in self.contents:
            if len(self.contents[group_id]) > self.size:
                self.contents[group_id] = self.contents[group_id][1:self.size]
            self.contents[group_id].append(content)
        else:
            self.contents[group_id] = [content]

    def to_string(self, group_id):
        arr = []
        for item in self.contents[group_id]:
            arr.append(item.to_string())
        return str(arr) # .replace("\"", "").replace("\'", "")

    def clear(self, group_id):
        self.contents[group_id]=[]

    def auto_set_weight(self, group_id):
        probabilities = np.random.uniform(0, 1, size=len(self.contents[group_id]))
        for i in range(0, len(self.contents[group_id])):
            self.contents[group_id][i].weight = probabilities[i]

