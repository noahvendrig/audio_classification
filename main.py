# Load various imports 
import pandas as pd
import librosa
import struct
import os
import numpy as np
import librosa.display

# Set the path to the full UrbanSound dataset 
fulldatasetpath = "D:/Datasets/UrbanSound8K/"
os.chdir(fulldatasetpath)

def extract_features(file_name):
   
    try:
        if os.path.exists(file_name):
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
        else:
            print("File not found "+file_name)
            
    except Exception as e:
        print(e)
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled
    
featuresdf = None
SAVE_FEATURES = False


if SAVE_FEATURES:
    metadata = pd.read_csv(fulldatasetpath + 'metadata/UrbanSound8K.csv')
    #metadata = metadata[0:10]

    features = []

    # Iterate through each sound file and extract the features 
    for index, row in metadata.iterrows():
        
        file_name = os.path.join(os.path.abspath(fulldatasetpath),'audio/fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        print(file_name)

        class_label = row["class"]
        data = extract_features(file_name)
        
        features.append([data, class_label])

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')

    # Write to file features.csv
    featuresdf.to_csv(fulldatasetpath + 'features.csv') 

else:
    # read from file features.csv
    featuresdf = pd.read_csv(fulldatasetpath + 'features.csv')
    print("Loaded "+ "features from features.csv")

# Converting the data and labels then splitting the dataset

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
l = featuresdf.feature.tolist()
alt = [x.replace('\n','').replace('  ',' ').replace('[','').replace(']','').split(' ') for x in l]
parsedFeatures = []
for row in alt:
    arr = [float(x) for x in row if len(x)>0]
    if len(arr)!=40:
        print("not 40: ", row)
    parsedFeatures.append(arr)
#X = np.array(featuresdf.feature.tolist())
X = np.array(parsedFeatures)
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

# Building the model
#

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

num_rows = 40
#num_columns = 174

num_channels = 1

#x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
#x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()

USE_PREVIOUS_WEIGHTS = True
checkpoint_path = fulldatasetpath+'/saved_models/weights.best.basic_cnn.hdf5'

TRAIN = False

if USE_PREVIOUS_WEIGHTS:
    # Loads the weights
    model.load_weights(checkpoint_path)

if TRAIN:


    # Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100*score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy) 

    from tensorflow.keras.callbacks import ModelCheckpoint 
    from datetime import datetime 

    num_epochs = 1000
    num_batch_size = 256

    checkpointer = ModelCheckpoint(filepath=fulldatasetpath+'/saved_models/weights.best.basic_cnn.hdf5', 
                                verbose=1, save_best_only=True)
    start = datetime.now()

    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)


    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

else:
    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    def print_prediction(file_name):
        prediction_feature = extract_features(file_name) 
        prediction_feature = prediction_feature.reshape(1, num_rows)

        print(prediction_feature.shape)
        predicted_vector = model.predict_classes(prediction_feature)
        predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", predicted_class[0], '\n') 

        predicted_proba_vector = model.predict_proba(prediction_feature) 
        predicted_proba = predicted_proba_vector[0]
        for i in range(len(predicted_proba)): 
            category = le.inverse_transform(np.array([i]))
            print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )


    # Class: Air Conditioner

    filename = fulldatasetpath+'testing_audio/'+'carhorn.wav' 
    print_prediction(filename)
    



