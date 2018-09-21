#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 02:08:34 2018

@author: celdel
"""

import face_recognition
import os
from pickle import dump,load
import pandas as pd 

data_dir = "data/images"
landmarks_file = "landmarks_file.b"
ff_file = "ff_file.b"
ff_file_csv = "ff_file.csv"

l = os.listdir(data_dir)

"""
l1 = []
for img_name in l:
    img_a = os.path.join(data_dir,img_name)
    #print(type(face_recognition.face_landmarks(face_recognition.load_image_file(img_a))))
    l1.append(face_recognition.face_landmarks(face_recognition.load_image_file(img_a)))
 """   
  
l1 = l1    
"""
dfg=open(landmarks_file,'wb')
dump(l1,dfg)
dfg.close()

l1_ff = []
for l1_obj in l1:
    l1_ff_obj = []
    for key in l1_obj[0]:
        for item in l1_obj[0][key]:
            l1_ff_obj.append(item[0])
            l1_ff_obj.append(item[1])
    l1_ff.append(l1_ff_obj)
"""

l1_ff=l1_ff
"""
dfg=open(ff_file,'wb')
dump(l1_ff,dfg)
dfg.close()
"""

df = pd.DataFrame(l1_ff)
df.to_csv(ff_file_csv)

