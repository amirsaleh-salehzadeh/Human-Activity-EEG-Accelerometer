import numpy as np
import band_filter

SampFreq = 256
ChannelNum = 22

def get_band_filter(data, st_feq, ed_feq):
    '''
    input: [channel, SampFreq]

    return: [channel, SampFreq] filterd at st_feq to ed_feq
    '''
    out_data = []
    for i in range(0, ChannelNum):
        channel_data = data[i]
        filterd_channel_data = band_filter.butter_bandpass_filter(
                channel_data, st_feq, ed_feq, SampFreq)
        out_data.append(filterd_channel_data)
    return out_data


def make_single_image(data):
    '''

    input: [channel, SampFreq]
    return [3, channel, SampFreq]

    '''
    R_data = get_band_filter(data, 0.01, 7)
    G_data = get_band_filter(data, 8, 13)
    B_data = get_band_filter(data, 13, 30)
    image = []
    image.append(R_data)
    image.append(G_data)
    image.append(B_data)
    return image

def make_image_string(data, time_step):
    '''
    input: data [sample, channel, feature]
    output image data [sample, timestep, 3, channel, sampfreq]
    '''
    image_data = []
    new_data = []
    for item in data:
        new_sample = []
        for i in range(0, time_step):
            time_data = []
            for channel in item:
                new_channel = channel[i*SampFreq:(i+1)*SampFreq]
                time_data.append(new_channel)
            new_sample.append(time_data)
        new_data.append(new_sample)
    print(np.shape(new_data))
    # new data: [sample, timestep, channel, feature]


    count = 1
    for sample in new_data:
        image_sample = []
        print('process {}-th image', count)
        count += 1
        for t_win in sample:
            image = make_single_image(t_win) 
            image_sample.append(image)
        image_data.append(image_sample)
    print(np.shape(image_data))
    return image_data


def main():
    pass

if __name__ == '__main__':
    main()
