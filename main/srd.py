'''
Created on Mar 31, 2019

@author: dulan
'''
from params import FILES
from data.data_loader import data_loader
from main.preprocess.singlish_preprocess import singlish_preprocess
from main.lstm_model import lstm_model


def main():
    print("Running the main program...")
    data_loader_obj = data_loader()
    singlish_preprocess_obj = singlish_preprocess()
    lstm_obj = lstm_model()

    x, y = data_loader_obj.load_data_csv(FILES.CSV_FILE_PATH)
    # print(x[25])
    # print(singlish_preprocess_obj.pre_process(x[25]), len(singlish_preprocess_obj.pre_process(x[25])))
    X = [singlish_preprocess_obj.pre_process(xi) for xi in x]
    Y = [ 1 if yi == 'Racist' else 0 for yi in y]
    # print(X)
    # print(Y)
    lstm_obj.train(X, Y)

if __name__ == "__main__":
    main()

