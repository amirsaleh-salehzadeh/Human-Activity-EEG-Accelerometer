
'''
为每个患者在每个网络（shallowconv、deepconv）上跑50个epoch 得到val结果
'''
import os
import subprocess
import datetime

#run_list = [6, 8, 12, 13, 14, 15]
run_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24]
print(run_list)
run_script = 'eeg-lstm/train.py'
start_epoch = 0
epoch = 50
d = datetime.date.today()
date_prefix = '{}_{}'.format(d.month, d.day)
save_root = './training_{}'.format(date_prefix)
model_name = 'shallowconv'
if not os.path.exists(save_root):
    os.mkdir(save_root)

for item in run_list:
    folder = './LOPO_image_slide_3/chb{0:02}'.format(item)
    save_dir = './training_{0}/model_chb{1:02}'.format(date_prefix, item)
    print(folder, save_dir)
    log_file = './{}.log'.format(item)
    with open(log_file, 'w') as f:
        subprocess.call('nohup python {} -f {} -s {} -se {} -e {} -m {} & '
            .format(run_script, folder, save_dir, start_epoch, epoch, model_name), shell=True, stdout=f, stderr=f)
