
# coding: utf-8

# In[1]:




# In[2]:

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

imgae_load = load_img('cat.0.jpg')  
imgae_load = img_to_array(imgae_load)  
imgae_load = imgae_load.reshape((1,) + imgae_load.shape)  


i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='datagen', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  


# In[ ]:



