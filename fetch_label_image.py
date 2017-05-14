#coding: utf-8

import os
import pandas as pd
import shutil

label_num = 24
each_image_num = 5

data = pd.read_csv(os.path.join('train_master.tsv'), sep='\t')
data_part = [data[data['category_id'] == i][:each_image_num] for i in range(label_num)]
for dp in data_part:
    for i, row in enumerate(dp.iterrows()):
        old_name = row[1]['file_name']
        new_name = '{0:02d}'.format(row[1]['category_id']) + '_' + str(i) + '.jpg'
        shutil.copyfile(os.path.join('train', old_name), os.path.join('image_example', new_name))