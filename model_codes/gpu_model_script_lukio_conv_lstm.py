# -*- coding: utf-8 -*-
"""
Created on  Jul 16 8:05:33 2018
8 layers
"""
#CNNs apply a series of filters to the raw pixel data of an image to extract and learn higher-level features,
import scipy,keras,PIL,time,os,io
import numpy as np
import fnmatch
from PIL import Image
import PIL
import keras.preprocessing.image as image
from keras.applications.vgg16 import preprocess_input
from keras.models import model_from_json
import tensorflow as tf
#Backend Settings
from keras import backend as K
#if K.backend()=='tensorflow':K.set_image_dim_ordering("th")
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib

#check the number of gpu devices available 
print(device_lib.list_local_devices())

#check the 
#nvidia-smi
#os.environ["CUDA_DEVICE_OR_DER"]= "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #INSERT the gpu-device id returned by print(device_lib.list_local_devices())
#List CPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']
#List GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
name = get_available_gpus()
print("GPUS:{}".format(name))
G= len(name)#nr of gpus available

#--------------------------------------------------------Data Preparation---------------------------------------------------------------
data_dir='D:/DataProphet/data'
model_results_dir='D:/DataProphet/model_results'
os.chdir(data_dir) #contains data from 01/01/2012 to 12/07/2018 ,2385 images in total
listOfFiles = os.listdir('.')  

            
#load images and turn them into np arrayy
datas = []
#%%timeit #21 s per loop
for img_name in listOfFiles:
    img= image.load_img(img_name)
    img=img.crop(box=(27,12,371,237)) #crop the images to remove legend & unwanted borders
    img_array= image.img_to_array(img)#image with shape (320, 400, 3)
    img_array= img_array/255
    datas.append(img_array)
#img=img.reshape((1,)+img.shape)# (1, 320, 400, 3) add dimension
datas_array=np.array(datas) #shape=(2385, 225, 344, 3)

sample_size=50 
time_steps= 7
img_height=225 
img_width=344
nr_channels=3
datas_array_inuse= datas_array[2285:2385,:,:,:]#use the last 107 daily maps
train_data = np.zeros((sample_size, time_steps,img_height, img_width,nr_channels))##samples==nr of images in online learning: batch learning require too much memory
test_data_shifted = np.zeros((sample_size, time_steps,img_height, img_width,nr_channels))
print(np.shape(train_data),np.shape(test_data_shifted)); 

#x=np.expand_dims(x,axis=0
#fill training and shifted_test_data, generate 100 samples
for i in range(50):
    print(i)
    train_data[i:,:,:,:,:] = datas_array_inuse[i:i+time_steps] 
    test_data_shifted[i:,:,:,:,:] = datas_array_inuse[i+1:i+1+time_steps]
#test_data_shifted=test_data_shifted[:,1:4,:,:,:]  #3 frames for shifted data  
np.savez(os.path.join(model_results_dir,'big_processed_prec_data')  , X=train_data,y=test_data_shifted) #data for training
np.savez(os.path.join(model_results_dir,'big_datas_array_inuse'), whole_data_used=datas_array_inuse) #main data from which all other used in other parts  can be extracted




#----------------Forecasting and Learning-------------------------------
import numpy as np
import keras 
from keras.models import Sequential
from keras import layers
from keras.layers import Conv3D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
import h5py



#def main():
#load the final array 
npz_dta = np.load(os.path.join(model_results_dir,'big_processed_prec_data'+'.npz'))
train_data = npz_dta['X']
shifted_test_data = npz_dta['y']
nr_filters = 40#the  data has 16 distinct precipitation classes, this informs the nr of hidden states we pursue.


#with tf.device('/cpu:0'):
    #define your model that has been pinned on the cpu device
lukio_precip_model = Sequential()
lukio_precip_model.add(ConvLSTM2D(filters=nr_filters, kernel_size=(5, 5),#try with a larger kernel
                       input_shape=(None,225,344,3),name="conv2d_lyr1",
                       padding='same', return_sequences=True,activation="elu"))#strides=7
lukio_precip_model.add(BatchNormalization())

#model.add(Dropout(0.25))
lukio_precip_model.add(ConvLSTM2D(filters=nr_filters, kernel_size=(5, 5),padding='same',name="conv2d_lyr2", return_sequences=True,activation="relu"))
lukio_precip_model.add(BatchNormalization())
#lukio_precip_model.model.add(MaxPooling2D(pool_size=(5, 5), padding='same'))

lukio_precip_model.add(ConvLSTM2D(filters=nr_filters, kernel_size=(5, 5),padding='same', name="conv2d_lyr3",return_sequences=True,activation="relu"))
lukio_precip_model.add(BatchNormalization())

lukio_precip_model.add(Conv3D(filters=3, kernel_size=(5,5,5), padding='same', name="conv3d_lyr4",activation="relu"))

# replicate the model on 6 gpus, or check the  nr of gpus available to your server provision
lukio_precip_model_parallel= multi_gpu_model(lukio_precip_model, gpus=G)#ASSUMES there is atleast more than 1 gpu as set in the beginning of this script

# initialize the optimizer and model
adam=keras.optimizers.adam(lr=0.1)
lukio_precip_model_parallel.compile(loss='mean_squared_error', optimizer='adam',metrics=["mse"])
#lukio_precip_model.compile(loss='mean_squared_error', optimizer=RMSprop,metrics=["mse"])

# # train the net
lukio_precip_model_parallel.fit(train_data[:,2:6,:,:,:],shifted_test_data[:,1:5,:,:,:],epochs=20, validation_split=0.2,batch_size=5)

#transfer weight to the main model
lukio_precip_model.compile() #
lukio_precip_model.set_weights(lukio_precip_model_parallel.get_weights()) # transfer the trained weights from GPU model to base model
lukio_precip_model.save(os.path.join(model_results_dir,'my_precip_conv_lstm6.h5'))


# serialize model to JSON for Visualization Libraries
model_json = lukio_precip_model.to_json()
with open(os.path.join(model_results_dir,'my_precip_model6'+'.json'), "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
lukio_precip_model.save_weights(os.path.join(model_results_dir,'my_precip_model_weights6'+'.h5'))#Saved model to disk

##############################-----------------------Forecasting ---------------------------------
#os.chdir(model_results_dir)
os.chdir("D:/DataProphet/model_results")
# load json and create model
json_file = open('D:/DataProphet/model_results/my_precip_model6.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("D:/DataProphet/model_results/my_precip_model_weights6.h5")

len(loaded_model.get_weights())
loaded_model.weights
#------------------------------------Sequence_by_Sequence Predictions using Normalised Frames---------
#-----Data :  Normalised
#prepare train_dataset and shifted_dataset as described in Data Preparation Section in the PDF, 
#the script that includes data preparation is attached for reproducing this step
#pick the last 8 images(supposedly not used in predictions) to help in forecasting.
whole_data_inuse = np.load("D:/DataProphet/model_results/datas_array_inuse.npz")
whole_data_inuse=whole_data_inuse['whole_data_used']#100 frames

    
#take 7 frames to use for predicting 13,ft+1
seven_input_frames_N=whole_data_inuse[87:100,::,::,::] #10 frames, forecasting target=7, =>17
X_input_Norm = seven_input_frames_N[0:10,:,:,:] #th
X_save= np.zeros((1,225,344,3))



#sequence-by-sequence prediction: 
for j in range(0,10):
    print("Shape of X_input1: ",X_input_Norm.shape)
    new_predictions= loaded_model.predict(X_input_Norm[np.newaxis, ::, ::, ::, ::])#returns a sequence of images of shape (nr_seq,h,w,channel)
    print("Shape of new_predictions: ",new_predictions.shape)
    if j == 0:
        X_save = new_predictions[:,2,:]
    else:
        X_save=np.concatenate((X_save,new_predictions[:,2,:]),axis=0) 
    print("Shape of X_shave:",X_save.shape)
    X_input_Norm=np.concatenate((X_input_Norm[np.newaxis,:], new_predictions), axis=0)
    print("Shape of X_input2: ",X_input_Norm.shape)
    X_input_Norm=X_input_Norm[1:]
    X_input_Norm=X_input_Norm[0,:]

#save the prediction frames
np.savez('D:/DataProphet/model_results/prediction_images/predicted_frames',prediction_frames=X_save) #data for training 
pred_frames = np.load("D:/DataProphet/model_results/prediction_images/predicted_frames.npz")
#prepare dates
listOfFiles = os.listdir('.')  
pred_frames=listOfFiles[2375:2385]#the last 10 dates,3  used in validation, 7 are time-ahead forecasts
validation_dates=[]
for k in pred_frames:
    str_dates=k.replace(".gif","")
    xx=pd.to_datetime(str_dates)
    xx=xx.strftime('%Y-%m-%d')
    validation_dates.append(xx)
    
#xx=pd.to_datetime(pred_frames[-1].replace(".gif",""))#the last_date in the input dataset
#next_seven_dates_forecasted= pd.date_range(start=pred_frames[-1].replace(".gif",""), periods=7)

#-------plot ouput-----------
#from matplotlib import rcParams
#rcParams['axes.titlepad']=20

font= {
    'family':'serif',
    'color':'darkred',
    'weight':'normal',
    'size':12
}

fig, ((ax1, ax2,ax3), (ax4, ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3)
for i in range(1,10):
    grob=eval("ax"+str(i))
    x1=X_save[i,:]/np.max(X_save[i,:])
    #plt.imshow(x1,cmap="Wistia_r");plt.colorbar();plt.title("Precipitation Map "+ validation_dates[i])
    grob.imshow(x1,cmap="Wistia_r");grob.set_title(validation_dates[i],fontsize=10);
    grob.axes.set_xticklabels(labels=[1,2,3,4,5,6,7],visible=False);grob.axes.set_yticklabels(labels=[1,2,3,4,5,6,7],visible=False)
    map_img= image.array_to_img(x1)
    image.save_img(x=map_img,path=os.path.join(os.getcwd(),"image_forecast_"+validation_dates[i]+".jpeg"))
fig.suptitle("Precipitation Forecast for US",fontsize=16)
fig.tight_layout(rect=[0,0.03,1,0.9])
fig.text(0.1,0.005,'Validation Sets: Plots '+validation_dates[1]+'&'+validation_dates[2],fontdict=font,wrap=True)
fig.set_alpha(0.9)
fig.savefig('precipitation_forecast.png',edgecolor='blue',orientation='landscape')
#fig.subplots_adjust(right=0.8);
#cb_ax= fig.add_axes([0.85,0.15,0.05,0.9])
#plt.imshow(X_save[1,:],cmap="Wistia_r")
#plt.colobar(cax=cb_ax)
#cbar.set_ticks(0.01,1.5,3,5,8,10,12)
#cbar.set_ticklabels([0.01,1.5,3,5,8,10,12])

#-----------------------------------Model Evaluation Using New Image Frames-----------------------------------------------------

# evaluate loaded model on test data, with train_data and shifted_data preprared as at the beginning of this script
#I could use cross-entropy loss but in a sense  I am not classifying images.
loaded_model.compile(loss='mean-squared-error', optimizer='adam', metrics=['mse'])
score = loaded_model.evaluate(train_data, shifted_data, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))





