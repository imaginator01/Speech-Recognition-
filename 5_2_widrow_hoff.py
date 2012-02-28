#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def calc_error(weight_vector_set,training_data_set,class_num):
    result=0.0
    for class_id in range(class_num):
        w = weight_vector_set[class_id]
        for training_data in training_data_set:
            x = training_data.x
            if class_id == training_data.class_id:
                b_ip = 1
            else:
                b_ip = 0
            result += np.power(np.dot(w,x) - b_ip, 2)
    return result / 2.0

def update_weight(weight_vector_set,training_data_set,raw):
    for class_id in range(class_num):
        w = weight_vector_set[class_id]
        diff_v = np.array([0.0,0.0])
        for training_data in training_data_set:
            x = training_data.x
            if class_id == training_data.class_id:
                b_ip = 1
            else:
                b_ip = 0
            diff_v += (np.dot(w,x) - b_ip) * x
        w -= raw * diff_v

    return weight_vector_set

class TrainingData():
    def __init__(self, class_id, x1):
        self.class_id = class_id
        self.x = np.array([1,x1])

#training_data_dict = {0:[0.5,1.0],1:[-1.3,-0.2]}
training_data_dict = {0:[-0.4,0.5,1.0],1:[-2.0,-1.3,-0.2]}

training_data_set=[]
for class_id,vals in training_data_dict.items():
    for val in vals:
        training_data_set.append(TrainingData(class_id,val))

w0 = [0.2,0.3]
raw=0.2

weight_vector_set=[]
class_num = 2
for i in range(class_num):
    weight_vector_set.append(np.array(w0))

num = len(training_data_set)

loop_ctr=0
success_flg=0

J = calc_error(weight_vector_set,training_data_set,class_num)
print "init_J:%f"%J

# 終了条件をどうするか?
#    (1) weight_vectorを更新してもJがbetterにならない場合
#    (2) Jがしきい値より小さくなった場合
#    (3) Jの更新幅がしきい値より小さくなった場合
#
#  --> (3)とする。

threshold = 0.001

while(loop_ctr < 20):

    loop_ctr+=1
    print "\nloop_ctr:%d"%loop_ctr

    w_v_s_new = update_weight(weight_vector_set,training_data_set,raw)

    J_new = calc_error(weight_vector_set,training_data_set,class_num)
    print "J_new:%f"%J_new
    if (np.abs(J_new-J) < threshold):
        print "\t converged."
        break
    else:
        J = J_new
        weight_vector_set = w_v_s_new

print "final_J:%f"%J

for i in range(class_num):
    print "\nw_%d:"%(i)
    print weight_vector_set[i]
