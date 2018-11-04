import numpy as np
import os
import make_image
import combine_data

ChannelNum = 22
SampFreq = 256

def load_data(x, y):
    data = np.load(x)
    label = np.load(y)
    return data, label

#data.shape:[[[], [], [], ...], [[], [], [], ...], ...]  sample * channel * points 
def gen_image(data):
    image_data = []
    count = 0
    for item in data:
        #image.shape: [[[], [],...], [[], [],...], [[], [],...]] [3, channel, points]
        image = make_image.make_single_image(item)
        image_data.append(image)
        print('generating {} image'.format(count))
        count += 1
    #image_data.shape: [[[[], [],...], [[], [],...], [[], [],...]], [[[], [],...], [[], [],...], [[], [],...]], ...] [sample, 3, channel, points]
    return image_data

def main():
    origindata_dir = './raw_slide_3'
    out_dir = './image_slide_3'
#
#    for dirpath, dirnames, filenames in os.walk(origindata_dir):
#        structure = os.path.join(out_dir, dirpath[len(origindata_dir):][1:])
#        print(structure)
#        if not os.path.isdir(structure):
#            os.mkdir(structure)
#        else:
#            print("Folder does already exits!")
#    folder_list = list(range(1, 25))
#    folder_list.remove(16)
#    print(folder_list)
#    for folder in folder_list:
#        out_dir_name = os.path.join(out_dir, 'chb{0:02}'.format(folder))
#        in_dir_name = os.path.join(origindata_dir, 'chb{0:02}'.format(folder))
#        for file in os.listdir(in_dir_name):
#            if '.npy' in file:
#                data = np.load(os.path.join(in_dir_name, file))
#                image_data = gen_image(data)
#                np.save(os.path.join(out_dir_name, file), image_data)
#
    file_list = []
    for file in os.listdir(origindata_dir):
            if 'data' in file:
                file_list.append(os.path.join(origindata_dir, file))

    for file in file_list:
        basename = os.path.basename(file)[:5]
        print('generating cnn set for {}'.format(basename))
        data_path = file
        label_path = combine_data.get_label_file_path(data_path)
        o_data, o_label = load_data(data_path, label_path)
        data = gen_image(o_data)

        np.save(os.path.join(
            out_dir, '{}_data.npy'.format(basename)), data)

        np.save(os.path.join(
            out_dir, '{}_label.npy'.format(basename)), o_label)




if __name__ == '__main__':
    main()
