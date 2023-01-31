#!/usr/bin/env python
# coding: utf-8

# ### importamos las librerias a utilizar

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os


# ### Preparacion de espacio de trabajo
# 
# especificamos la ruta de nuestro dataset, un arreglo para guardar las categorias que vayamos encontrando y la dimension a trabajar con nuestras imagenes

# In[8]:


DATADIR = '/home/lenin/Documents/datasets/colores'
CATEGORIES = []
IMG_SIZE=28


# recorremos el direcotorio excluyendo las imagenes de testeo

# In[9]:


for cate in os.listdir(DATADIR):
    if cate == 'testimg': continue
    CATEGORIES.append(cate)
print(f'categorias encontradas: {CATEGORIES}\n total: {len(CATEGORIES)}')


# ### Recoleccion de datos
# recorremos el directorio tomando cada imagen segun su respectiva categoria

# In[10]:


training_data=[] #var para los datos recolectados
labels=0
for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    for fname in os.listdir(path):
        img = load_img((path+'/'+fname), target_size=(IMG_SIZE,IMG_SIZE))
        x = img_to_array(img)
        x=x/255
        training_data.append([x,labels])
    labels+=1
print('done')


# verificamos el total de datos que tenemos

# In[12]:


lenofimage = len(training_data)
print(lenofimage)


# ### Tratamiento de los datos
# separamos  nuestra data en img-labels y los tratamos con numpy

# In[14]:


X=[]
y=[]
for img, label in training_data:
    X.append(img)
    y.append(label)
print('done')


# verificamos la forma de nuestra data

# In[15]:


X=np.array(X)
y=np.array(y)
print('labels',y.shape)
print('img',X.shape)


# dividimos la data para train y test

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X,y)
print(f'train: {len(X_train)}, test: {len(X_test)}')


# ### Creacion del modelo
# creamos la arquitectura del modelo, lo compilamos y finalmente ajustamos

# In[17]:


model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='SAME', input_shape=X_train[0].shape),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(15, activation='relu'),
    Dense(len(CATEGORIES), activation='softmax'),
])


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=20)


# verificamos la prescicion en testeo

# In[20]:


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('test loss',test_loss)
print('test accuracy',test_accuracy)


# ### Probamos el modelo entrenado

# In[23]:


#ruta a nuestras imagenes de test en el directorio, no de la data preparada
path = DATADIR + '/testimg' + '/49.jpg' 

#tratamos la img con el mismo tamanio y la normalizamos
img = load_img(path, target_size=(IMG_SIZE,IMG_SIZE))
x = img_to_array(img)
x=x/255

#agrego un eje para que el modelo lo reciba
x = x[np.newaxis, ...]

#imprimo la img y la prediccion
plt.imshow(img)
resp = model.predict(x)
print(f'prediccion = {CATEGORIES[np.argmax(resp[0])]}')


# ### Exportacion
# con el modelo ya funcionando correctamente, lo exportamos para poder utilizarlo en otros proyectos

# In[24]:


model.save('mod_color_v1.h5')


# In[ ]:




