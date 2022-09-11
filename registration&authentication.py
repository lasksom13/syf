import json
import sys
import os

import numpy as np

from keystrokes_recorder import record
from model_and_data_preparation_handler import ModelSetup
from sklearn.metrics.pairwise import cosine_similarity

class ClientHandler:

    def __init__(self):
        # Assign to self
        self.model, self.central_vector = ModelSetup.load_model_from_dir()

    def registration(self):
        dict = self.load_json_file_to_dict()
        while True:
            login = input('Type your login: ')
            if login in dict.keys():
                print("{} is already taken. Choose different login.".format(login))
            else:
                break
        print("rewrite this 5 times: \033[92m\033[1m.tie5roanl \x1b[0m press Enter after each time")
        samples = []
        iteration = 1
        while iteration < 6:
            print(f"({iteration}): ", end="")
            sample = record()
            if len(sample) == 22:
                samples.append(sample[:20])
                iteration += 1
            else:
                print("     ", len(sample))

        temp = np.empty((20, 0))
        samples = np.array(samples).T
        temp = np.concatenate((temp, samples[:,:]), axis=1)
        temp = np.array(temp).T
        temp = np.expand_dims(temp, axis=-1)
        output = self.model.predict(temp)
        output_vector = np.mean(output, axis=0)
        dict[login] = (np.subtract(output_vector, self.central_vector)).tolist()
        self.save_dict_as_json_file(dict)
        return

    def authentication(self):
        dict = self.load_json_file_to_dict()
        while True:
            login = input('Type your login: ')
            if login in dict.keys():
                break
            else:
                print("Login {} is not in clients.json file. Please register or choose different login".format(login))

        print("Please write this: \033[92m\033[1m.tie5roanl \x1b[0m and press Enter")
        samples = []
        while True:
            print("(1): ", end="")
            sample = record()
            if len(sample) == 22:
                samples.append(sample[:20])
                break
            else:
                print("     ", len(sample))

        print(samples[0])
        temp = np.empty((20, 0))
        samples = np.array(samples).T
        temp = np.concatenate((temp, samples[:, :]), axis=1)
        temp = np.array(temp).T
        temp = np.expand_dims(temp, axis=-1)
        output = self.model.predict(temp)

        # print(type(output))
        # print(output.shape)
        print(output)
        sum: float = 0
        for element in output[0,:]:
            print(element)
            sum += element
        print(sum)

        output_vector = np.mean(output, axis=0)
        output_vector_2 = np.subtract(np.mean(output_vector, axis=0), self.central_vector)
        # print(output_vector)
        # print(output_vector_2)
        self.check_score(dict, output_vector_2, login)
        return

    def load_json_file_to_dict(self):
        try:
            with open("clients.json", 'r') as f:
                data = json.load(f)
            return data
        except IOError as error:
            print("Loading json has failed")
            print(error)
            sys.exit(1)

    def save_dict_as_json_file(self, dict):
        try:
            json_object = json.dumps(dict, indent=4)
            with open("clients.json", 'w') as f:
                f.write(json_object)
        except IOError as error:
            print("Saving dict as json file has failed")
            print(error)
            sys.exit(1)

    def check_score(self, dict, sample, owner_login):
        list_of_score_tuple = []
        for login in dict.keys():
            template = np.array(dict[login])
            score = cosine_similarity(np.expand_dims(template, axis=0), np.expand_dims(sample, axis=0))
            list_of_score_tuple.append( (score, login) )

        sorted_by_score = sorted(list_of_score_tuple, key=lambda tuple: tuple[0], reverse=True)
        for tuple in sorted_by_score:
            if tuple[1] == owner_login:
                print('\033[92m\033[1m', tuple, '\x1b[0m')
            else:
                print(tuple)
        return

if __name__ == '__main__':
    instance_ClientHandler = ClientHandler()
    # instance_ClientHandler.registration()
    instance_ClientHandler.authentication()
    sys.exit()


