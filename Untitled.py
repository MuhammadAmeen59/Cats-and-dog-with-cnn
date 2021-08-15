#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow


# In[8]:


from tensorflow.keras.datasets import mnist
(train_images , train_labels),(test_images , test_labels)=mnist.load_data()


# In[9]:


train_images.shape


# In[10]:


train_labels.shape


# In[12]:


train_images[0]


# In[18]:


digit1=train_images[93]
import matplotlib.pyplot as plt
plt.imshow(digit1, cmap=plt.cm.binary)
plt.show()


# In[22]:


from tensorflow.keras import models
from tensorflow.keras import layers
network=models.Sequential()
network.add(layers.Dense(512,activation="relu", input_shape = (28 * 28,)))
network.add(layers.Dense(10,activation="softmax"))


# In[23]:


network.compile(optimizer="rmsprop" , loss="categorical_corssentropy",metrics=["accuracy"])


# In[ ]:




