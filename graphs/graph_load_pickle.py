import pickle
from config import *


# LOAD THE DATA, PICKLE FORMAT
def load_data_from_pickle(file_path):

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

# load the data
pretrained_loss = load_data_from_pickle('pretraining_loss.pkl')

finetuning_train_loss = load_data_from_pickle('finetuning_train_loss.pkl')
finetuning_train_acc = load_data_from_pickle('finetuning_train_accuracy.pkl')

finetuning_val_loss = load_data_from_pickle('finetuning_val_loss.pkl')
finetuning_val_acc = load_data_from_pickle('finetuning_val_accuracy.pkl')

benchmark_train_loss = load_data_from_pickle('benchmark_train_loss.pkl')
benchmark_train_acc = load_data_from_pickle('benchmark_train_accuracy.pkl')

benchmark_val_loss = load_data_from_pickle('benchmark_val_loss.pkl')
benchmark_val_acc = load_data_from_pickle('benchmark_val_accuracy.pkl')
