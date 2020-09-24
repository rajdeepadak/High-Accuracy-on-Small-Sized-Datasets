#%% Import all libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import Callback, ModelCheckpoint


#%% Initialise CNN
classifier = Sequential()

#%% Input Layer
# Step 1-a Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu', ))
# Step-1b Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#%% Hidden Layer 1
# Step-2a Convolution
classifier.add(Conv2D(48, (3, 3), activation = 'relu' ))
# Step-2b Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#%% Output Layer
# Step-4 Flattening
classifier.add(Flatten())


#%% Step-5 Full Connection
classifier.add(Dense(28, input_shape = (64, 16), activation = 'relu'))
classifier.add(Dropout(0.4))

classifier.add(Dense(14, input_shape = (64, 8), activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(1, activation = 'sigmoid'))


#%% Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#%% Constructing the image data generator for Train set and test set
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.3,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/Train_Set',
        target_size=(128, 128),
        batch_size=64,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/Test_Set',
        target_size=(128, 128),
        batch_size=64,
        class_mode='binary')


#%% Fitting the CNN into image data, beginning training

class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='val_accuracy', value=0.86, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            Warning.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


callbacks = [
    EarlyStoppingByAccuracy(monitor='val_accuracy', value=0.86, verbose=1),
    ModelCheckpoint(filepath='kfold_weights_path.h5', monitor='val_accuracy', save_best_only=True, verbose=0),
]


history = classifier.fit_generator(
        training_set,
        steps_per_epoch=204,
        epochs=10,
        callbacks=callbacks,
        validation_data=test_set,
        validation_steps=52)


#%% Plotting accuracy results
import matplotlib.pyplot as plt 

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.title('input layer = 32, hidden layer = 48')

#%% Import necessary Libraries

import os 
import cv2
import numpy as np
from keras.preprocessing import image 

#%% Resize Function
def Resize(img):
    # Resize Image to 480x360
    resize = cv2.resize(img, (480, 360))
    return resize

#%% Gamma Correction Function
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
 
#%% Prediction on a Set of Images randomly taken from the internet 

# Test on a set of random images
print("List of sub folders available for testing : \n1. Test_NL \n2. Test_AL \n3. Random")

print("\n\nNote: \n1. Test_NL consists of all images with naturally lit rooms or rooms which definitely have\
 some natural light. \n2. Test_AL has all images with only artificially lit rooms. \n3. Random has 5 images\
 of naturally lit rooms and 5 images of only artificially lit rooms. \n\nYou may add more images in these files\
 to test as desired.")

i = int(input("Enter choice from above list on which prediction test is to be performed : "))

fold_list = ['Test_NL', 'Test_AL', 'Random']

folder_loc = 'C:\\Lighting_Classifier\\Dummy_Image_Test\\'


for subdir, dirs, files in os.walk('C:\\Lighting_Classifier\\' + fold_list[i-1]):
    for filename in files:
        filepath = subdir + os.sep + filename
        
        pred_NL = []
        pred_AL = []

        if filepath.endswith(".jpg"):
            
            img = cv2.imread(filepath)
            
            for j in np.arange(0.02, 0.36, 0.02):    
                g_img = adjust_gamma(img, gamma = j)
                file_path = folder_loc + str(j) + '.jpg'
                cv2.imwrite(file_path, Resize(g_img))
                
            for j in np.arange(0.5, 0.6, 0.04):    
                g_img = adjust_gamma(img, gamma = j)
                file_path = folder_loc + str(j) + '.jpg'
                cv2.imwrite(file_path, Resize(g_img))
            
            
            for subdir_, dirs_, files_ in os.walk(r'C:\Lighting_Classifier\Dummy_Image_Test'):
                for filename_ in files_:
                    filepath_ = subdir_ + os.sep + filename_
            
                    if filepath_.endswith(".jpg"):
                        
                        img_i = cv2.imread(filepath_)
                        n = 0
                        sum_ = 0
                        
                        for l in img_i:
                            for m in l:
                                for n in m:
                                    sum_ = sum_ + n
                        
                        
                        if sum_ < 2700000 or sum_ > 40000000:
                            try: 
                                os.remove(filepath_)                                
                            except: pass
                        
                        else:
                            test_image = image.load_img(filepath_, target_size = (128, 128))
                            test_image = image.img_to_array(test_image)
                            test_image = np.expand_dims(test_image, axis = 0)
                
                            result = classifier.predict(test_image)            
                            
                            # training_set.class_indices
                            
                            if result[0][0] == 1:
                                prediction = 'Natural Light'
                                pred_NL.append(1)
                                
                                
                            else:
                                prediction = 'Artificial Light'
                                pred_AL.append(0)
                          
            AL_per = (len(pred_AL) / (len(pred_AL) + len(pred_NL)))*100
            NL_per = (len(pred_NL) / (len(pred_AL) + len(pred_NL)))*100            
                        
            print('Image in', filepath, 'is', AL_per, '% Artificial Lighting and', NL_per,'% Natural Lighting')
            
            