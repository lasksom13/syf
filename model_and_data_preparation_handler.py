import numpy as np
import pandas as pd
import os.path
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Input, Dropout
from sklearn.metrics.pairwise import cosine_similarity
from neural_network_plots import Neural_Network_Performance
import sys
import json


class ModelSetup(Neural_Network_Performance):

    def __init__(self):
        # Assign to self object
        self.model_summary = None
        self.model_history = None
        self.train_data = None
        self.eval_data = None
        # Action to execute
        self.setup()
        self.evaluate()

    def setup(self):
        # read Killourhy dataset
        df = ModelSetup.read_csv_file()
        # remove unnecessary data then split remaining data into train_set & eval_set
        self.train_data, self.eval_data = ModelSetup.split_data(df)
        # model setup + model training&validation + central_vector
        X, X_train, X_valid, Y_train, Y_valid = ModelSetup.prepare_data_for_model_training_and_validation(self.train_data)
        model = self.model_layers()
        self.model_training_and_validation(model, X_train, Y_train, X_valid, Y_valid, X)

    def evaluate(self):
        # Load model and central_vector
        model, central_vector = ModelSetup.load_model_from_dir()
        # Create enroll templates and test samples
        enroll, test = self.enrollment(model, central_vector)
        # Start cross-evaluation process
        confidence_TP_MLP, confidence_TN_MLP = ModelSetup.cross_evaluation(enroll, test)
        # Draw plots and save them in plots file
        ModelSetup.confidence_figure(confidence_TP_MLP, confidence_TN_MLP)
        ModelSetup.model_accuracy_figure(self.model_history)
        # ModelSetup.DETcurve_figure(confidence_TP_MLP, confidence_TN_MLP)

    @staticmethod
    def read_csv_file(filename: str = 'DSL-StrongPasswordData.csv'):
        df = pd.read_csv(filename, sep=',')
        df = np.array(df)
        return df

    @staticmethod
    def split_data(df):
        df = np.delete(df, [0, 1, 2], axis=1)
        df = np.delete(df, [-1, -2], axis=1)
        df = np.delete(df, list(range(2, df.shape[1], 3)), axis=1)
        train_data = {}
        eval_data = {}
        for person in range(41):
            temp = []
            for sample in range(0, 400):
                temp.append(df[person * 400 + sample][:])
            temp = np.squeeze(np.array(temp))
            temp = temp.astype(float)
            train_data[person] = temp.T

        for person in range(41, 51):
            temp = []
            for sample in range(0, 400):
                temp.append(df[person * 400 + sample][:])
            temp = np.squeeze(np.array(temp))
            temp = temp.astype(float)
            eval_data[person] = temp.T

        return train_data, eval_data

    @staticmethod
    def prepare_data_for_model_training_and_validation(train_data: dict):
        X = np.empty((20, 0))
        Y = []
        for it, person_id in enumerate(train_data.keys()):
            X = np.concatenate((X, train_data[person_id]), axis=1)
            for i in range(0, 400):
                Y.append(it)
        X = np.array(X).T
        Y = np.array(Y)
        Y_onehot = to_categorical(Y, num_classes=41)
        X = np.expand_dims(X, axis=-1)
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_onehot, test_size=0.2, random_state=123)
        return X, X_train, X_valid, Y_train, Y_valid

    def model_layers(self):
        model = Sequential()
        model.add(Input(shape=(20, 1)))
        model.add(LSTM(units=10))
        model.add(Dense(10, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(15))
        model.add(Dense(41, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model_summary = model.summary()
        return model

    def model_training_and_validation(self, model: Sequential, X_train, Y_train, X_valid, Y_valid, X):
        self.model_history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=600, batch_size=60)
        central_vector = np.mean(model.predict(X), axis=0)
        # save model
        ModelSetup.save_model_and_central_vector(model, central_vector)

    @staticmethod
    def save_model_and_central_vector(model: Sequential, central_vector, directory: str = "model") -> None:
        if os.path.exists(directory):
            model.save(directory)
            with open("model/central_vector.pickle", "wb") as file:
                pickle.dump(central_vector, file, protocol=pickle.HIGHEST_PROTOCOL)

    # methods for evaluation
    @staticmethod
    def load_model_from_dir(directory: str = "./model") -> (Sequential, np.ndarray):
        # CHECK IF THE FOLDER EXISTS
        if os.path.exists(directory):
            model = load_model(directory)
            with open(directory + "/central_vector.pickle", "rb") as file:
                central_vector = pickle.load(file)
            return model, central_vector

    def enrollment(self, model, central_vector):
        enroll = {}
        test = {}
        enroll_sample_number = 5
        test_samples_per_person = 60
        for person_id in self.eval_data.keys():
            # Enroll
            temp = np.empty((20, 0))
            temp = np.concatenate((temp, self.eval_data[person_id][:, 0:enroll_sample_number]), axis=1)
            temp = np.array(temp).T
            temp = np.expand_dims(temp, axis=-1)
            output = model.predict(temp)
            out_vector = np.mean(output, axis=0)
            enroll[person_id] = np.subtract(out_vector, central_vector)
            # Generate pseudo users (temporary solution for testing)
            self.some_other_users(person_id,enroll[person_id])
            # Sample
            test[person_id] = []
            for sample in range(enroll_sample_number, enroll_sample_number + test_samples_per_person):
                temp = self.eval_data[person_id][:, sample:sample+1]
                temp = np.array(temp).T
                temp = np.expand_dims(temp, axis=-1)
                output_temp = model.predict(temp)
                test[person_id].append(np.subtract(np.mean(output_temp, axis=0), central_vector))

        return enroll, test

    @staticmethod
    def cross_evaluation(enroll, test):
        confidence_TP_MLP = []
        confidence_TN_MLP = []

        for userA in enroll.keys():
            userA_model = np.expand_dims(enroll[userA], axis=0)
            for sample in test[userA]:
                confidence_TP_MLP.append(cosine_similarity(userA_model, np.expand_dims(sample, axis=0)))

            for userB in test.keys():
                if userB != userA:
                    for sample in test[userB]:
                        confidence_TN_MLP.append(cosine_similarity(userA_model, np.expand_dims(sample, axis=0)))

        confidence_TP_MLP = np.squeeze(np.array(confidence_TP_MLP))
        confidence_TN_MLP = np.squeeze(np.array(confidence_TN_MLP))

        return confidence_TP_MLP, confidence_TN_MLP


    def some_other_users(self, username, vector):
        dict = self.load_json_file_to_dict()
        dict[username] = vector.tolist()
        self.save_dict_as_json_file(dict)
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
            return
        except IOError as error:
            print("Saving dict as json file has failed")
            print(error)
            sys.exit(1)


if __name__ == '__main__':
    ins_ModelSetup = ModelSetup()







