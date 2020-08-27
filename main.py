import numpy as np

from lstm import LSTM
from neural_networks import Artificial_Neural_Network as Fully_Connected
from get_data import questions, results, confirmations

class LSTMxFC:
    def __init__(self, input_size, LSTMoutputsize=64, LSTMhiddensize=64):
        self.lstm = LSTM(input_size, LSTMoutputsize, LSTMhiddensize)
        self.answers = []
        self.FC = dict()
    
    def add(self, key, network: Fully_Connected):
        self.FC[key] = network
    
    def add_answer(self, answer):
        self.answers.append(answer)
    
    def get_answers(self):
        return self.answers
    
    def compile_lstm(self):
        out = self.lstm.forward(np.array(np.array(self.answers)))
        self.output = out
        return out
    
    def clear_answers(self):
        self.answers.clear()
    
    def replace_answer(self, answers: list):
        self.answers = answers.copy()
    
    def forward_result(self, result_key):
        self.compile_lstm()
        out = self.FC[result_key].forward(self.output)
        return out
    
    def compile_all_results(self):
        out = dict()
        for key in self.FC.keys():
            out[key] = self.forward_result(key)
        return out
    
    def train(self, targets: dict, learning_rate: float):
        out = self.compile_all_results()
        LSTM_loss = 0
        LSTM_MSE = 0
        FC_loss = 0
        for k in out.keys():
            dL_dy = out[k] - targets[k]
            FC_loss += np.sum(dL_dy ** 2)
            dL_dLSTM = self.FC[k].backprop(dL_dy, learning_rate, alpha=0)
            LSTM_loss += dL_dLSTM
            LSTM_MSE += dL_dLSTM ** 2
        self.lstm.backprop(LSTM_loss, learning_rate)
        return np.mean(FC_loss), np.mean(LSTM_MSE)

default_hidden_layers = [64, 32, 64]

def show_question(lstm: LSTMxFC, questions):
    for question in questions:
        print(question)
        answer = int(input("Enter Answer: "))
        lstm.add_answer(question.choice_to_onehot(answer))

def get_target(questions):
    amswers = []
    for question in questions:
        print(question)
        answer = int(input("Enter Answer: "))
        amswers.append(question.choice_to_onehot(answer))
    return amswers

choice_size = 4
lstmxfc = LSTMxFC(choice_size)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
d_sigmoid = lambda y: y * (1 - y)
for result in results:
    lstmxfc.add(result.result_key, Fully_Connected(64, len(result.result_value), default_hidden_layers, sigmoid, d_sigmoid))

if __name__ == "__main__":
    XY = []
    while True:
        show_question(lstmxfc, questions)
        res = lstmxfc.compile_all_results()
        ruh = [np.argmax(r) + 1 for r in res]
        actual_result = get_target(confirmations)
        print("-" * 30)
        print("Your result")
        for i in range(len(ruh)):
            print(results[i](ruh[i]))
        keys = ["Depression", "Anxiety", "Stress"]
        actual_result_ = {}
        for i in range(len(actual_result)):
            actual_result_[keys[i]] = np.array(actual_result[i])
        (X, Y) = (lstmxfc.get_answers().copy(), actual_result_.copy())
        XY.append([X, Y])
        fcloss = 0
        for x, y in XY:
            lstmxfc.replace_answer(x)
            fcloss += lstmxfc.train(y, 0.1)[0] / len(XY)
        print()
        print("Current Loss: {}".format(fcloss))
        print("What is loss ? It's the value that tells how far from the accurate target !")
        print("-" * 30)
        if input("Are you willing to take a test again ? (Y / N): ") == "N":
            break
