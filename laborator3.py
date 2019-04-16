
# coding: utf-8

# # Rezolvare laborator 3

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataPath = "C:\\Documente\\anulII\\SemII\\IA\\data\\"
#load train images
train_images = np.loadtxt(dataPath + "train_images.txt")
test_images = np.loadtxt(dataPath + "test_images.txt")
# print(train_images.shape)
# print(train_images.ndim)
# print(type(train_images[0,0]))
# print(train_images.size)
# print(train_images.nbytes)


# In[3]:


#plot the first image
image = train_images[0,:]
image = np.reshape(image,(28,28))
# plt.imshow(image,cmap = "gray")
# plt.show()


# In[4]:


#load training labels
train_labels = np.loadtxt(dataPath + "train_labels.txt",'int8')
test_labels = np.loadtxt(dataPath + "test_labels.txt",'int8')
# print(train_labels.shape)
# print(type(train_labels[0]))
# print(train_labels.size)
# print(train_labels.nbytes)
# #show the label of the first training image
# print(train_labels[0])


# In[5]:


#plot the first 100 training images with their labels in a 10 x 10 subplot
nbImages = 10
# plt.figure(figsize=(5,5))
# for i in range(nbImages**2):
#     plt.subplot(nbImages,nbImages,i+1)
#     plt.axis('off')
#     plt.imshow(np.reshape(train_images[i,:],(28,28)),cmap = "gray")
# plt.show()
# labels_nbImages = train_labels[:nbImages**2]
# print(np.reshape(labels_nbImages,(nbImages,nbImages)))

#laborator3
pC=np.zeros(10)
for i in range(1000):
    c=train_labels[i]
    pC[c]=pC[c]+1
print(pC)
pC.argmax()
#pentru clasa 7 pozitita 370
C=7
poz=370
indicii=np.ravel(np.where(train_labels==7))
print(indicii)
print (len(indicii))
print(train_labels[:10])
values=train_images[indicii,poz]
print(len(values))
nbBins=4
bins=np.linspace(0,256,nbBins+1)
print(bins)
h=np.histogram(values,bins)
print(h)
hh=h[0]/sum(h[0])
print(hh)

#antrenare
#pentru toate clasele si toate pozitiile
M=np.zeros((10,784,nbBins)) #matrice tridimesionala M[7,370,:]=hh
for c in range(10):
    indicii = np.ravel(np.where(train_labels == c))
    for poz in range(784):
        values = train_images[indicii, poz]
        h = np.histogram(values, bins)
        M[c,poz,:]=h[0]/sum(h[0])+0.000000001
print(M[7,370,:])

#tesatare
img=test_images[0]
logpc=np.log(pC)
#imparte [0,255] in 4 intervale de 64 si da intervalul pt valorile din values
values=np.array([0,255,23,60,75,130])
b=np.digitize(values,bins)-1
print(b)
for poz in range(784):
    value=img[poz]
    b=np.digitize(value,bins)-1
    for c in range(10):
        logpc[c]+=np.log(M[c,poz,b])
print(logpc.argmax())
plt.imshow(np.reshape(img,(28,28)), cmap="gray")
plt.show()

#pentru toate imaginile
#matricea de confuzie 10x10, cate cifre lin sunt confundate cu cifre col
