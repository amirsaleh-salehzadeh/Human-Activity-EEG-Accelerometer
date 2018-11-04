import os
import argparse
import csv
import pprint
def get_parser():
    parser = argparse.ArgumentParser(description='gen models overview')
    parser.add_argument('-f', '--folder', help='models dir')
    return parser.parse_args()


def get_models_info(models_path):
    folder_list = []
    for folder in os.listdir(models_path):
        if os.path.isdir(os.path.join(models_path, folder)):
            folder_list.append(os.path.join(models_path, folder))

    models_overview = []
    for folder in folder_list:
        #find training log
        log_path = ''
        for file in os.listdir(folder):
            if 'training' in file:
                log_path = os.path.join(folder, file)
        index = os.path.basename(folder)
        acc, best_loss, epoch = get_best_loss_model(log_path)
        models_overview.append({'model':index, 'acc': acc, 'loss':best_loss, 'epoch':epoch})
    pprint.pprint(models_overview)
    
    max_record, avg_record = get_maxAvg_loss_model(models_overview)
    pprint.pprint('the max loss record is : {}'.format(max_record))
    pprint.pprint('the average acc and loss : {}'.format(avg_record))
        
def get_best_loss_model(log_path):
    #按loss最小求得最好的epoch
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        lowest_loss = 1024
        index = []
        for each in data_list[1:]:
            epoch = each[0]
            val_loss = float(each[4])
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                index = each
        acc = float(index[3])
        best_loss = float(index[4])
        epoch = index[0]
        return acc, best_loss, epoch

def get_maxAvg_loss_model(models_overview):
    max_loss = 0
    max_record = {}
    avg_record = {'acc': 0, 'loss':0}
    count = 0
    for each_model in models_overview:
        loss = each_model['loss']
        acc = each_model['acc']
        model = each_model['model']
        epoch = each_model['epoch']
        if loss > max_loss:
            max_loss = loss
            max_record = {'model':model, 'acc': acc, 'loss':loss, 'epoch':epoch}
            
        avg_record['acc'] += acc
        avg_record['loss'] += loss
        count += 1
    avg_record['acc'] = avg_record['acc']/count
    avg_record['loss'] = avg_record['loss']/count
    return max_record, avg_record

def main():
    args = get_parser()
    models_path = args.folder
    get_models_info(models_path)
    
    
if __name__ == '__main__':
    main()
