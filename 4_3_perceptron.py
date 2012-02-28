#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class TrainingData():
    def __init__(self, class_id, x1):
        self.class_id = class_id
        self.x = np.array([1,x1])

training_data_dict = {1:[0.5,1.0],2:[-1.3,-0.2]}

training_data_set=[]
for class_id,vals in training_data_dict.items():
    for val in vals:
        training_data_set.append(TrainingData(class_id,val))

w0 = [0.2,0.3]
raw=0.5

w=np.array(w0)
num = len(training_data_set)

loop_ctr=0
success_flg=0

while(loop_ctr < 5):
    loop_ctr+=1
    print "\nloop_ctr:%d"%loop_ctr

    classification_count=0

    for training_data in training_data_set:
        x = training_data.x
        x_vector = x.tolist()
        w_vector = w.tolist()
        print "\t x : (%f,%f)"%(x_vector[0],x_vector[1])
        print "\t w : (%f,%f)"%(w_vector[0],w_vector[1])

        g_x = np.dot(w,x)
        print "\t\t class_id:%d, g_x:%f"%(training_data.class_id,g_x)

        if training_data.class_id==1 and g_x <= 0:
            w += raw * x
        elif training_data.class_id==2 and g_x >= 0:
            w -= raw * x
        else:
            classification_count+=1

        weight_vector = w.tolist()
        print "\t w_new is [%f,%f])\n"%(weight_vector[0],weight_vector[1])

    if classification_count == num:
        success_flg=1
        break

if success_flg:
    weight_vector = w.tolist()
    print "successfully converged! (loop_ctr:%s,weight_vector is [%f,%f])"%(loop_ctr,weight_vector[0],weight_vector[1])
else:
    print "failed to converge! (loop_ctr:%s,wrong_classification:%s)"%(loop_ctr,num-classification_count)
