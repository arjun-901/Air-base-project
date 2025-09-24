# src/train_all.py
import multiprocessing
from src.train_one_model import train_one_model

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    model_names = ['ann', 'cnn', 'lstm', 'vgg9', 'vgg16', 'encoder_decoder']
    processes = []

    for model_name in model_names:
        process = multiprocessing.Process(target=train_one_model, args=(model_name,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
