#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import numpy as np

# 演習問題6-5 neural net by back_propagation

# 3層のneural networkで考える。

# input layer (I)
# intermediate layer (M)
# output layer (O)


# (学習データpに関する)重みの更新式
#    w_ij -= raw * (epsilon_jp * g_ip)

# 各unitにおける誤差の変化量 epsilon_jp の表式

# (1) @ output layer
#        (g_jp - b_jp) * g_jp * (1 - g_jp)

# (2) @ intermediate layer
#        (Σ_k epsilon_kp * omega_jk) * g_jp * (1 - g_jp)

# sigmoid function
#    s(u) = 1 / (1+exp^(-u))



# algo.

# - 外部から教師信号と学習係数を与える
# - 適当に重み行列の初期値を決める
# for training_data in training_dataset:
#    -- 入力信号とA_IMをもとに中間層からの出力信号を計算
#    -- 中間層からの出力信号とA_MOをもとに、最終的な出力信号を計算
#    -- output layerの重み行列を更新
#    -- intermediate layerの重み行列を更新

# 終了条件
#    案1 : 一定回数、上記loopを回す
#    案2 : 重みの調整量がしきい値以下になる

# local minにはまるのを避けるため、一般には、異なる初期値から複数回学習を試みるらしい

def sigmoid_func(x):
	#print "sigmoid:%f"%(x)
	#return 1.0 if x>0 else 0.0
	return 1.0 / (1 + math.exp(-x))

class TrainingData():
    def __init__(self, class_id, input_list):
        self.class_id = class_id
        self.val_vector = np.array(input_list)


# - 外部から教師信号と学習係数を与える

#training_data_dict = {0:[[1,4],[2,3],[4,3],[5,4]],1:[[2,1],[3,2],[3,3],[4,1]]}
training_data_dict = {0:[[1.0/5.0,4.0/4.0],[2.0/5.0,3.0/4.0],[4.0/5.0,3.0/4.0],[5.0/5.0,4.0/4.0]],1:[[2.0/5.0,1.0/4.0],[3.0/5.0,2.0/4.0],[3.0/5.0,3.0/4.0],[4.0/5.0,1.0/4.0]]}
#training_data_dict = {0:[[1.0/5.0,4.0/4.0],[2.0/5.0,3.0/4.0],[4.0/5.0,3.0/4.0],[5.0/5.0,4.0/4.0]],1:[[2.0/5.0,1.0/4.0],[3.0/5.0,2.0/4.0],[4.0/5.0,1.0/4.0]]}

training_data_set=[]
for class_id,vals in training_data_dict.items():
    for val in vals:
        training_data_set.append(TrainingData(class_id,[0.5] + val))

class_num = 2
raw=0.01

# 重み行列
#    A_IM (d * m)
#    A_MO (m * c)

A_IM = np.zeros((3, 5))
for i,row in enumerate(A_IM):
	for j, w_ij in enumerate(row):
		A_IM[i,j] =random.uniform(-1.0,1.0)

A_MO = np.zeros((5, 2))
for i,row in enumerate(A_MO):
	for j, w_ij in enumerate(row):
		A_MO[i,j] =random.uniform(-1.0,1.0)

loop_ctr=0
success_flg=0
threshold = math.pow(10,-4)

while(loop_ctr < 10000 and success_flg == 0):

	loop_ctr+=1
	print "\nloop_ctr:%d"%loop_ctr

	error_val=0.0

	for training_data in training_data_set:

		# 入力信号とA_IMをもとに中間層からの出力信号を計算
		mediate_signal = np.array(map((lambda x: sigmoid_func(x)),np.dot(training_data.val_vector,A_IM).tolist()))

		# 中間層からの出力信号とA_MOをもとに、最終的な出力信号を計算
		output_signal = map((lambda x: sigmoid_func(x)),np.dot(mediate_signal,A_MO).tolist())

		#print "mediate_signal",mediate_signal
		#print "output_signal",output_signal

		# output layerの重み行列の仮更新
		A_MO_new = A_MO.copy()

		epsilon_vector = []
		for j in range(class_num):
			g_jp = output_signal[j]
			b_jp = 0.9 if training_data.class_id == j else 0.1

			print "g_jp:%f,b_jp:%f"%(g_jp,b_jp)

			epsilon_jp = (g_jp - b_jp) * g_jp * (1 - g_jp)
			epsilon_vector.append(epsilon_jp)

		for i,row in enumerate(A_MO):
			for j, w_ij in enumerate(row):
				g_ip = mediate_signal[i]
				A_MO_new[i,j] -= raw * g_ip * epsilon_vector[j]
				error_val += np.abs(g_ip * epsilon_vector[j])

		# intermediate layerの重み行列を更新
		for i,row in enumerate(A_IM):
			for j, w_ij in enumerate(row):
				g_ip = (training_data.val_vector)[i]
				g_jp = mediate_signal[j]

				sum_of_epsilon = 0
				for k in range(class_num):
					sum_of_epsilon += epsilon_vector[k] * A_MO[j,k]
				A_IM[i,j] -= raw * g_ip * sum_of_epsilon * g_jp * (1 - g_jp)

				#error_val += np.abs(g_ip * sum_of_epsilon * g_jp * (1 - g_jp))

		# output layerの重み行列の本更新
		A_MO = A_MO_new

	if( error_val < threshold):
		success_flg=1

	print "error_val:%f"%error_val
	#print "A_IM:",A_IM
	#print "A_MO:",A_MO


'''
所感 @ 2012/3/11(Sun.)

	上記実装中のsigmoid関数はstep関数にしちゃうとまずいんだろうなあ、やっぱ・・・。
	sigmoid関数の勾配を急にする様に関数形のparameterをいじってみる?

	この学習データだと難しすぎ?、と思って、微妙なデータである、(3,3)を除いたデータセットで
	走らせて見たけど、大して結果は変わらなかった・・・。


	特徴ベクトルの0次元目としてx_0=1をいれとく意味は何? (この実装ではやっていないけど)
		--> 識別関数において、特徴ベクトルによらず、加えておきたい「下駄」としての重みを表現するために、
		便宜的に、入れている。

		--> ってことはこれも実装に取り込まないと駄目だね・・・。
		--> 取り込んでみた。
		--> あんまし効果なし。

	最終結果でも、
		出力信号ベクトル (g_0,g_1) の成分間の大小が、
		(本来なら、正解ラベルに応じて逆転して欲しいのに)

		常に g_0 >g_1 という傾向が見られた。


	重みの更新処理のところに問題がある??

	それとも、もう少しbasicなところで間違っている?


所感 @ 2012/3/12(Mon.)
	主な変更点
		- 特徴ベクトルの値が全次元で0～1になる様にした
		- x_0を0.5にした
		- 重み行列の初期値は各セルで-1～1内のrandom値とした
		- output layerの重み行列の更新は、intermediate layerの重み行列の更新後とした
			-- 後者の修正量算出時に、前者の値が効いてくる為
		- 教師信号ベクトルを、(正解ノードだけ0.9、他は0.1)となる様にした( 1/0 だと、sigmoid関数の出力値との差がいつまでも残る可能性があるので)
		- 誤差としては、全学習データに渡る累積値を採用する様にした
			-- どっかの学習データにて、誤差がしきい値以下だったらloop終了、というのはちょっと微妙だった為

	動かしての所感
		- text記載のむずかしげな学習データセットだと、1000回、走らせても収束しない。
		- 一方、微妙なデータである、(3,3)を除いたデータセットで走らせると、1000回後にて、割とよさげな出力信号となった
			-- 出力信号ベクトル (g_0,g_1) の成分間の大小が、正解ラベルに応じて逆転している

		- 中間層のノード数を振ってみたが、そんなに差はなかった。
		- あとは学習係数の値ぐらいかなあ・・・。

	参考 : http://tercel-sakuragaoka.blogspot.com/2012/02/processing_20.html

'''