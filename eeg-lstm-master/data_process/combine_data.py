import numpy as np
import os
from shutil import copyfile

def get_label_file_path(data_file_path):
    basename = os.path.basename(data_file_path)
    dir_name = os.path.dirname(data_file_path)
    label_basename = basename.replace('data', 'label')
    return os.path.join(dir_name, label_basename)

def combine(file_list):
    data_combine = []
    label_combine = []
    for data_file in file_list:
        label_file = get_label_file_path(data_file)
        data = np.load(data_file)
        label = np.load(label_file)
        data_combine.extend(data)
        label_combine.extend(label)
    #data_combine.shape: [sample, 3, channel, points]
    return data_combine, label_combine



def combine_data(data_dir, out_dir):
    '''
    simply combine all data in data_dir to one set
    '''
    file_list = []
    for file in os.listdir(data_dir):
            if 'data' in file:
                file_list.append(os.path.join(data_dir, file))
    data_combine, label_combine = combine(file_list)
    print(np.shape(data_combine), np.shape(label_combine))

    np.save(
            os.path.join(out_dir, 'data.npy'), data_combine)
    np.save(
            os.path.join(out_dir, 'label.npy'), label_combine)

def generate_LOPO_data(data_dir, out_dir):
    '''
    generate Leave one patient out data based on data_dir
    will generate a training set and a validation set, 
    the validation set come from the left patient
    '''
    file_list = []
    data_combine = []
    label_combine = []
    for file in os.listdir(data_dir):
            if 'data' in file:
                file_list.append(os.path.join(data_dir, file))

    for patient in file_list:
        # genrating LOPO data for patient
        basename = os.path.basename(patient)[:5]
        data_dir = os.path.join(out_dir, basename)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        print('generating LOPO data for {}'.format(basename))
        LOPO_list = [i for i in file_list if i != patient]
        data_name = '{}-data-train.npy'.format(basename)
        label_name = '{}-label-train.npy'.format(basename)
        data, label = combine(LOPO_list)
        np.save(
                os.path.join(data_dir, data_name), data)
        np.save(
                os.path.join(data_dir, label_name), label)
        # generate validation set
        data_name = '{}-data-val.npy'.format(basename)
        label_name = '{}-label-val.npy'.format(basename)
        copyfile(patient, os.path.join(data_dir, data_name))
        copyfile(
                get_label_file_path(patient), 
                os.path.join(data_dir, label_name))



def main():
    #combine_data('./raw_slide_3')
    generate_LOPO_data('./image_slice_1_s', './LOPO_image_slice_1_s')

if __name__ == '__main__':
    main()
