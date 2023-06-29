import tensorflow.keras.initializers
from tensorflow.keras.layers import  Dense,Input,Reshape,Conv2D,Flatten,MaxPooling2D,AveragePooling2D,Embedding,Dropout,GlobalAveragePooling1D,GlobalMaxPooling1D
from tensorflow.keras.models import Model
import keras
import importlib
import math
import tensorflow as tf
from tensorflow.keras import layers



def Sentiment():
    max_features = 10000
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim,embeddings_initializer = 'he_uniform'),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4,activation = tf.keras.activations.softmax)])
    return model
def Sentiment_zero():
    initializer = tensorflow.keras.initializers.Zeros()
    max_features = 10000
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim,embeddings_initializer = initializer),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4,activation=tf.keras.activations.softmax,kernel_initializer=  initializer)])
    return model


def CNN(shape,num_classes):
    input=Input(shape=shape)
    CNN1= Conv2D(6, (5, 5), activation='relu', kernel_initializer='he_uniform')(input)
    AVG1=AveragePooling2D()(CNN1)
    CNN2= Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_uniform')(AVG1)
    AVG2 = AveragePooling2D()(CNN2)
    flatten=Flatten()(AVG2)
    DENSE1=Dense(120, activation='relu')(flatten)
    DENSE2 = Dense(84, activation='relu')(DENSE1)
    predictions=Dense(num_classes, activation='softmax')(DENSE2)
    architecture = Model(inputs=input, outputs=predictions)
    return architecture

def CNN_zero(shape,num_classes):
    initializer = tensorflow.keras.initializers.Zeros()
    input=Input(shape=shape)
    CNN1= Conv2D(6, (5, 5), activation='relu', kernel_initializer=  initializer)(input)
    AVG1=AveragePooling2D()(CNN1)
    CNN2= Conv2D(16, (5, 5), activation='relu', kernel_initializer= initializer)(AVG1)
    AVG2 = AveragePooling2D()(CNN2)
    flatten=Flatten()(AVG2)
    DENSE1=Dense(120, activation='relu', kernel_initializer=  initializer)(flatten)
    DENSE2 = Dense(84, activation='relu', kernel_initializer=  initializer)(DENSE1)
    predictions=Dense(num_classes, activation='softmax', kernel_initializer=  initializer)(DENSE2)
    architecture = Model(inputs=input, outputs=predictions)
    return architecture



def CNN_VGG(shape,num_classes):
    input=Input(shape=shape)
    CNN1= Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input)
    CNN2=Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(CNN1)
    MAX1 = MaxPooling2D((2, 2))(CNN2)
    CNN3=Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(MAX1)
    CNN4=Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(CNN3)
    MAX2 = MaxPooling2D((2, 2))(CNN4)
    CNN5=Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(MAX2)
    CNN6=Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(CNN5)
    MAX3 = MaxPooling2D((2, 2))(CNN6)
    flatten=Flatten()(MAX3)
    DENSE1=Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    predictions=Dense(num_classes, activation='softmax')(DENSE1)
    architecture = Model(inputs=input, outputs=predictions)
    return architecture


def CNN_VGG_zero(shape,num_classes):
    initializer = tensorflow.keras.initializers.Zeros()
    input=Input(shape=shape)
    CNN1= Conv2D(32, (3, 3), activation='relu', kernel_initializer=  initializer, padding='same')(input)
    CNN2=Conv2D(32, (3, 3), activation='relu',  kernel_initializer=  initializer, padding='same')(CNN1)
    MAX1 = MaxPooling2D((2, 2))(CNN2)
    CNN3=Conv2D(64, (3, 3), activation='relu',  kernel_initializer=  initializer, padding='same')(MAX1)
    CNN4=Conv2D(64, (3, 3), activation='relu', kernel_initializer=  initializer, padding='same')(CNN3)
    MAX2 = MaxPooling2D((2, 2))(CNN4)
    CNN5=Conv2D(128, (3, 3), activation='relu',  kernel_initializer=  initializer, padding='same')(MAX2)
    CNN6=Conv2D(128, (3, 3), activation='relu',  kernel_initializer=  initializer, padding='same')(CNN5)
    MAX3 = MaxPooling2D((2, 2))(CNN6)
    flatten=Flatten()(MAX3)
    DENSE1=Dense(128, activation='relu',  kernel_initializer=  initializer)(flatten)
    predictions=Dense(num_classes, activation='softmax')(DENSE1)
    architecture = Model(inputs=input, outputs=predictions)
    return architecture

def return_model( dataset, x_shape, num_classes):
    if dataset == 'Sentiment':
        return Sentiment(), Sentiment(), Sentiment_zero()

    if dataset == 'CNN':
        return CNN(x_shape, num_classes), CNN(x_shape, num_classes), CNN_zero(x_shape,num_classes)

    if dataset == 'VGG':
        return CNN_VGG(x_shape, num_classes), CNN_VGG(x_shape, num_classes), CNN_VGG_zero(x_shape, num_classes)