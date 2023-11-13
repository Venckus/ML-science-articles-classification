import os
import pickle

class Models:
    def __init__(self) -> None:
        self.models = {}
        self.labels_dict = {
            'svm_computer_science': 'Computer Science',
            'svm_mathematics': 'Mathematics',
            'svm_physics': 'Physics',
            'svm_statistics': 'Statistics',
            'lr_biology': 'Quantitative Biology',
            'lr_finance': 'Quantitative Finance',
        }

    def load(self) -> None:
        for model in os.listdir('../models'):
            if model[:-4] in self.labels_dict:
                file = open('../models/' + model, 'rb')
                self.models[model[:-4]] = pickle.load(file)
                file.close()