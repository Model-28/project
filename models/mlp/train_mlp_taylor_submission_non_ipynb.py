'''
Not done as a notebook, let me know if you want me to change it to a notebook. This is what I finished over
the weekend. I'll be adding more visualizations of the model as well.
'''
import sys, os
# make sure project root is on sys.path



##----------------------
# this is for taylor's machine
# PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
##----------------------



##----------------------
# this is for sai's machine
# figure out the folder this script lives in
HERE = os.path.dirname(os.path.abspath(__file__))
# go up 3 levels to reach your `project/` folder (which contains data_preprocess/)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
##----------------------




from data_preprocess import preprocess
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
#from tensorflow.keras.utils import to_categorical

def main():
    raw_dir = os.path.join(PROJECT_ROOT, "..", "archive", "seg_train", "seg_train")
    # this print proves main() is running
    X_path,y_path = 'X.npy','y.npy'

    #check if npy files exist
    print("Checking if files exist...")
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("Not found, now preprocessing imgs")
        preprocess(
        input_dir=raw_dir,
        target_size=(150,150),
        out_X="X.npy",
        out_y="y.npy"
        )
        print("Finished preprocessing")
    else:
         print("Files found!")

    #Load datasets and get shapes
    X,y = np.load('X.npy'), np.load('y.npy')
    print("X shape:", X.shape, "y shape:", y.shape)

    #making the MLP, using ReLU then softmax
    print("Making model")
    model = Sequential()
    model.add(Dense(256, activation='relu',input_shape=(67500,))) #layer one, 256 features, relu
    model.add(Dense(128, activation='relu')) #layer two, 128 features from layer one, relu
    model.add(Dense(64,activation='relu')) #layer three, 64 features from layer two, relu
    model.add(Dense(len(np.unique(y)), activation='softmax')) #output layer, using features from y, softmax

    #compiling
    print("summarizing")
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) #measure via accuracy, loss is calculated

    # ——— Callbacks ———---------------------------------------------------------------------------
    # Stop early if val_loss doesn’t improve for 3 epochs, and restore the best weights
    es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate by half if val_loss plateaus for 2 epochs
    rlrop = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
    # Save the best‐ever model by val_accuracy
    mc = ModelCheckpoint(
        filepath="best_mlp.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    ## ------------------------------------------------------------------------------------------------------

    model.summary() #prints a summary of each layer, including number of parameters

    #splitting training and test data, 80:20
    print("Splitting data and training!")
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=42)
    # model.fit(Xtrain,ytrain,epochs=10,batch_size=30,validation_data=(Xtest,ytest))

    callbacks = [es, rlrop, mc]

    # changed fit to account for callbacks as well as flask requirements
    model.fit(
        Xtrain, ytrain,
        epochs=20,
        batch_size=30,
        validation_data=(Xtest, ytest),
        callbacks=callbacks
    )

    model = tf.keras.models.load_model("best_mlp.h5") ## saving the model into a file

if __name__=="__main__":
	main()

