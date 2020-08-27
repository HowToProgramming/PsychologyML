import json
import numpy as np

with open("psychotest.json", "r") as psychotest:
    psychotest = json.load(psychotest)

class Question:
    def __init__(self, question: dict):
        self.question_id = question['id']
        self.question = question['Question']
        self.choices = question['Choices']
    
    def __repr__(self):
        string = self.question_id + "\n" + self.question + "\n"
        for i, choice in enumerate(self.choices):
            string += str(i + 1) + ". " + choice + "\n"
        return string
    
    def choice_to_onehot(self, choice):
        zero = [[0]] * len(self.choices)
        zero[choice - 1] = [1]
        return np.array(zero)

class Result:
    def __init__(self, result_key, result_value):
        self.result_key = result_key
        self.result_value = result_value
    
    def __call__(self, idx):
        return self.result_value[idx]

questions = [Question(question) for question in psychotest['Questions']]
confirmations = [Question(question) for question in psychotest['Confirmation']]
results = [Result(key, value) for key, value in psychotest['Results'].items()]