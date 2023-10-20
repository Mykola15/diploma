# Ignore the warnings
import warnings
warnings.filterwarnings('ignore')

# data visualization and manipulation
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from keras.models import save_model
from PIL import Image  # You need to import Image from PIL library

# model selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# preprocess
from keras.preprocessing.image import ImageDataGenerator

# deep learning libraries
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# specifically for CNN
from keras.layers import Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import numpy as np
from tqdm import tqdm
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = []
Z = []
IMG_SIZE = 150
AIRPLANE_DIR = 'NEW_DATASET/AIRPLANE'
BIRD_DIR = 'NEW_DATASET/BIRD'
DRONE_DIR = 'NEW_DATASET/DRONE'
HELICOPTER_DIR = 'NEW_DATASET/HELICOPTER'
UAV_DIR = 'NEW_DATASET/UAV'

def make_train_data(data_type, folder):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'  # You can choose other fill modes as well
    )

    for img in tqdm(os.listdir(folder)):
        if img.endswith(".DS_Store"):
            continue
        path = os.path.join(folder, img)
        img = Image.open(path)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)

        # Check if the image is grayscale (2D)
        if len(img_array.shape) == 2:
            # Expand dimensions to (height, width, 3) for grayscale images
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1)  # Convert to 3 channels

        # Expand the dimensions to (1, IMG_SIZE, IMG_SIZE, 3) for data augmentation
        img_array = np.expand_dims(img_array, axis=0)

        # Generate augmented images
        aug_iter = datagen.flow(img_array, batch_size=1)

        # Append the augmented images and labels to X and Z
        for _ in range(2):  # Change the number to increase size
            augmented_img = next(aug_iter)[0]
            X.append(augmented_img)
            Z.append(str(data_type))

make_train_data('AIRPLANE', AIRPLANE_DIR)
make_train_data('BIRD', BIRD_DIR)
make_train_data('DRONE', DRONE_DIR)
make_train_data('HELICOPTER', HELICOPTER_DIR)
make_train_data('UAV', UAV_DIR)
print('Total Length: ', len(X))

fig, ax = plt.subplots(5, 2)
fig.set_size_inches(30, 30)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(Z))
        print("Selected index:", l)  # Add this line to check the selected index
        ax[i, j].imshow(X[l][0])
        ax[i, j].set_title('Object: ' + Z[l])

le = LabelEncoder()
Y = le.fit_transform(Z)
Y = to_categorical(Y, 5)
X = np.array(X)
X = X / 255

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

def train_knn_model(x_train, y_train, x_test, y_test):
    # Flatten the image data
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train_flatten, y_train)

    # Flatten the test data for prediction
    y_pred = model.predict(x_test_flatten)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def train_decision_tree_model(x_train, y_train, x_test, y_test):
    # Flatten the image data
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)

    model = DecisionTreeClassifier()
    model.fit(x_train_flatten, y_train)
    y_pred = model.predict(x_test_flatten)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def train_naive_bayes_model(x_train, y_train, x_test, y_test):
    # Flatten the image data
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)

    # Convert one-hot encoded labels back to class labels
    label_encoder = LabelEncoder()
    y_train_labels = label_encoder.fit_transform(np.argmax(y_train, axis=1))
    y_test_labels = label_encoder.transform(np.argmax(y_test, axis=1))

    model = GaussianNB()
    model.fit(x_train_flatten, y_train_labels)
    y_pred = model.predict(x_test_flatten)
    accuracy = accuracy_score(y_test_labels, y_pred)
    return model, accuracy

def train_random_forest_model(x_train, y_train, x_test, y_test):
    # Flatten the image data
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train_flatten, y_train)
    y_pred = model.predict(x_test_flatten)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def create_cnn_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation="softmax"))

    return model

def create_fnn_model(num_classes):
    model = Sequential()
    model.add(Flatten(input_shape = (IMG_SIZE,IMG_SIZE,3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_model(model, x_train, y_train, x_test, y_test, batch_size=128, epochs=50):
    # Learning rate annealer
    red_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.1)

    # Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False)

    datagen.fit(x_train)

    # Model compilation
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Model training
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  epochs=epochs, validation_data=(x_test, y_test),
                                  verbose=1, steps_per_epoch=x_train.shape[0] // batch_size,
                                  callbacks=[red_lr])

    # Specify the path where you want to save the model
    model_save_path = '/Users/mykola/Desktop/last_cnn_model.h5'

    # Save the model to the specified path
    save_model(model, model_save_path)

    return history

def show_results(val):
    plt.plot(val.history['loss'])
    plt.plot(val.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(val.history['accuracy'])
    plt.plot(val.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])
    plt.show()

knn_acc = train_knn_model(x_train, y_train, x_test, y_test)
print("K-Nearest Neighbors Accuracy:", knn_acc)

# Train Decision Tree model
decision_tree_acc = train_decision_tree_model(x_train, y_train, x_test, y_test)
print("Decision Tree Accuracy:", decision_tree_acc)

# Train Naive Bayes model
naive_bayes_acc = train_naive_bayes_model(x_train, y_train, x_test, y_test)
print("Naive Bayes Accuracy:", naive_bayes_acc)

# Train Random Forest model
random_forest_acc = train_random_forest_model(x_train, y_train, x_test, y_test)
print("Random Forest Accuracy:", random_forest_acc)

# Train and show results for the CNN model
cnn_model = train_model(create_cnn_model(5), x_train, y_train, x_test, y_test, 128, 50)
show_results(cnn_model)

# Train and show results for the FNN model
fnn_model = train_model(create_fnn_model(5), x_train, y_train, x_test, y_test, 128, 25)
show_results(fnn_model)
