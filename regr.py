import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from PIL import Image
import PIL
from sklearn.model_selection import train_test_split
from skimage import transform

import tensorflow as tf

from sklearn.metrics import r2_score


def model_train():
    global model
    df = pd.read_csv('data_sunglasses2.csv')
    df = df.sample(2850, random_state=1).reset_index(drop=True)
    train_df, test_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=1)
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='cena',
        target_size=(256, 256),
        color_mode='rgb',
        class_mode='raw',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='training'
    )
    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='cena',
        target_size=(256, 256),
        color_mode='rgb',
        class_mode='raw',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='validation'
    )
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='cena',
        target_size=(256, 156),
        color_mode='rgb',
        class_mode='raw',
        batch_size=32,
        shuffle=False
    )
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='mse'

    )
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True

            )
        ]
    )
    return model


model_train()
"""
# ten kod odpowiada za obliczenie sredniego bledu predykcji: Root mean square deviation
predicted_ages = np.squeeze(model.predict(test_images))
true_ages = test_images.labels

rmse = np.sqrt(model.evaluate(test_images,verbose=0))
print('Test RMSE {:.5f}'.format(rmse))

r2 = r2_score(true_ages,predicted_ages)
print('Test R^2 Score: {:.5f}'.format(r2))
# sredni blad wyniosl okolo 400zl
"""
#model.save('C:\\Users\\micpo\\PycharmProjects\\Zalando_Web_Scraping_Sunglasses\\my_model2')
uu = "C:\\Users\\micpo\\PycharmProjects\\Zalando_Web_Scraping_Sunglasses\\4626f842dd4a4f78b2487e2a3b71c16f.jpg"
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (256, 256, 3))
   np_image = np.expand_dims(np_image, axis=0)
   print("predicted price:",model.predict(np_image))
   return np_image

image = load('C:\\Users\\micpo\\PycharmProjects\\Zalando_Web_Scraping_Sunglasses\\4626f842dd4a4f78b2487e2a3b71c16f.jpg')

predicted_price = model.predict()

load(image)