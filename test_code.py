import os
import numpy as np
import math

PADDING = 20
SIZE = (10,6)
#Ron Shemesh's code.(Very messy)
#Global(fuck it)
final_training_data = []


def zero_padding(data):
	my_size = len(data)
	for i in range(PADDING - my_size):
		data.append(np.zeros(SIZE))
	return data

def find_nan(data):
	cnt = 0
	while cnt < data.shape[0]:
		if (math.isnan(float(data[cnt,0]))):
			return cnt
		cnt += 1
	return cnt

def normalize_pos(data):
	#Extract starting point.
	factor = np.array(data[0])
	factor[2] = 0
	factor[3] = 0
	factor[4] = 0
	factor[5] = 0
	#print("Data shape: %r \n",data.shape)
	for i in range(data.shape[0]):
		data[i,:] -= factor
	return data


'''Take the 30x6 matrix of training data and return a list of vectors of the size of 10
time specs(which is the min time specs of a rocket) 
Added size as input in order to be more generic.(it is 10 by default)'''
def divide_nums(data,size = 10):
	size_lst = []
	#Find first NaN.
	data_size = find_nan(data)
	#final_size = min(data_size - size + 1,data.shape[0] - size)
	for i in range(min(data_size - size + 1,data.shape[0] - size)):
		#print("It: %d \n" % i)
		curr_size = data[i:(i + size),:]
		curr_size = normalize_pos(curr_size)
		print("Current size: %r \n",curr_size.shape)
		size_lst += [curr_size]
		#print("Current LISTTTTTTTT: %r \n" % size_lst)
	size_lst = zero_padding(size_lst)
	return np.array(size_lst)

my_file = open("train_sample.csv",'r')
#Ignore first line.
f_dim = 0
train_data = []
y_train = []
for line in my_file.readlines()[1:]:
	#Filter \n.
	line = line.strip()
	#Count dimension.
	f_dim += 1
	curr_label = int(line[-1])
	#print("label: %r \n" % curr_label)
	#print("First line:%r \n" %line)
	#Take data of single rocket.Ignore first 2 examples of each line.
	curr_rocket = line.split(',')[2:]
	#Reshape list of data to 2D matrix,ignoring timespecs.
	#Ignore time_specs.
	#print("Before: %r \n" % curr_rocket)
	del curr_rocket[6:len(curr_rocket):7]
	#Convert to numpy arrays.
	curr_rocket = curr_rocket[:-1]
	curr_rocket = np.array(curr_rocket)
	#print("After: %r \n" % curr_rocket)
	curr_rocket = curr_rocket.reshape(30,6)
	#print("Rocket: %r \n" % curr_rocket)
	#Save rocket data in training data and label in y_train(vector of labels)
	train_data.append(curr_rocket)
	y_train.append(curr_label)

#Convert training data to numpy array form.
train_data = np.array(train_data)
y_train = np.array(y_train)
final_y = []

#Convert to floats.
train_data = train_data.astype(float)
y_train = y_train.astype(float)

test = divide_nums(train_data[0])
#print("Before: %r \n" % train_data[0])
#print("After %r \n" % test)
if y_train.shape[0] != train_data.shape[0]:
	print("ERROR")
'''After we got training data into 3D matrix and label vector,create training
	data of 10 specs each(this is the minimum),this is the way we filter all the
	NaN's (use max pooling later to filter the best part of the data.'''
for i in range(train_data.shape[0]):
	#Get the tens.
	#print("Iteration; %d \n" % i)
	curr_nums = divide_nums(train_data[i])
	print("Curr nums shape ",curr_nums.shape)
	final_training_data += [curr_nums]
	#final_y += curr_size * [y_train[i]] 

#print(final_training_data)
final_training_data = np.array(final_training_data)
print(final_training_data.shape)
train_file = open("train_numpy",'wb')
label_file = open("label_numpy",'wb')
np.save(train_file,final_training_data)
np.save(label_file,y_train)