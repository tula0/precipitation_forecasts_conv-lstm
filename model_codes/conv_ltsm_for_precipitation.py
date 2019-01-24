import scipy, keras,PIL,time,os,io
import numpy as np

#--------------------------------------------------------Data Preparation---------------------------------------------------------------
#load images and turn them into np arrayy
def read_file(file_name):

    with open(file_name) as f:
            img= load_img(f)
            img= img_to_array(img)#this is a numpy array ith shape (720, 1280, 3)
    return img


datas = []
with open('map_list', 'r') as file_list:
    for line in file_list.readlines():
        line = line.strip('\n')
        file_name=line
        #file_name = os.path.join(self.rawdir, line)
        d = read_file(file_name)
        datas.append(d)
      

img= load_img("D:/rosk.io(Working)/ML PRJ/SOM SPRJ/img3.jpg")
x= img_to_array(img)#this is a numpy array ith shape (720, 1280, 3)
x=x.reshape((1,)+x.shape)#this is a numpy array with shape (1, 720, 1280, 3)


#save and load the final array 
arch = np.load(self.saved_file_name)
train_data = arch['X']
shifted_test_data = arch['y']
np.savez(self.saved_file_name , X=self.train_data,y=self.shifted_data) #the data to be imported


#the shape of the data to be constructed is: (samples size,seq_length/time_steps,img_height,img_width,nr_of_channels(3 for RGB) )
time_steps=1251
train_data = np.zeros((samples="ntimes-nr_of_days_to_be_forecasted", timesteps_or_seqlrngth=24, img_h=70, img_w=60, nr_of_channes=3))##samples==nr of images in online learning: batch learning require too much memory
shifted_data = np.zeros((samples="ntimes-nr_of_days_to_be_forecasted", timesteps_or_seqlrngth=24, img_h=70, img_w=60, nr_of_channes=3))
print(np.shape(self.train_data),np.shape(self.shifted_data)); 


#fill training and shifted_test_data
for i in range(24):
    print(i)
    train_data[:,i,:,:,0] = datas_array[i:i+self.ntimes-24]
    shifted_data[:,i,:,:,0] = datas_array[i+1:i+1+self.ntimes-24]

## Create image batches
#the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely