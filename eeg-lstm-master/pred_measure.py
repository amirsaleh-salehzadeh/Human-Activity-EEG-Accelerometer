import numpy as np
from LOPO_train import load_data
import keras
import os
from sklearn.metrics import confusion_matrix



def load_model(model_path):
    return keras.models.load_model(model_path)


def get_cm(y_real, y_pred):
    conf_matrix = confusion_matrix(y_real, y_pred)
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    return TP, TN, FP, FN

def cal_sn(y_real, y_pred):
    TP, _, _, FN = get_cm(y_real, y_pred)
    sn = TP / (TP + FN) 
    return sn

def cal_sp(y_real, y_pred):
    _, TN, FP, _ = get_cm(y_real, y_pred)
    sp = TN / (TN + FP)
    return sp

def cal_bacc(y_real, y_pred):
    sn = cal_sn(y_real, y_pred)
    sp = cal_sp(y_real, y_pred)
    bacc = (sn + sp) / 2
    return bacc

def main():
    model_list = list(range(1, 24))
    model_list.remove(16)
    for model_num in model_list:
        model_dir = './model-ch{0:02}'.format(model_num)
        file_list = []
        lowest_loss = 2018
        model_file = ''
        for file in os.listdir(model_dir):
                if file.endswith(".hdf5"):
                    loss = float(file[-9:-5])
                    if loss < lowest_loss:
                        model_file = file
                        lowest_loss = loss
        model_path = os.path.join(model_dir, model_file)
        model = load_model(model_path)
        data_path = './LOPO/chb{0:02}'.format(model_num)
        test_data, test_label = load_data(
                os.path.join(
                    data_path, '{}-data-val.npy'.format(data_path[-5:])), 
                os.path.join(
                    data_path, '{}-label-val.npy'.format(data_path[-5:])))
        y_pred = model.predict(test_data)
        y_pred = np.argmax(y_pred, axis=1)
        print('best model of {}, path: {}'.format(model_num, model_path))
        print('sn: {}'.format(cal_sn(test_label, y_pred)))
        print('sp: {}'.format(cal_sp(test_label, y_pred)))
        print('bacc: {}'.format(cal_bacc(test_label, y_pred)))


if __name__ == '__main__':
    main()
