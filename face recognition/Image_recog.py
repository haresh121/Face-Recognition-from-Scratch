#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import PIL.Image as Image
import cv2
import sklearn
import matplotlib.pyplot as plt
from matplotlib import cm
import dlib
from imutils import face_utils
import os


# # PCA

# In[36]:


def PCA(A, b):
    if len(A.shape)>2:
        return ("Image has more tan 2 dims please input a correct photo")
    M = np.mean(A)
    A = A-M
    C = np.dot(A, A.T)
    val, vec = np.linalg.eigh(C)
    vec = np.dot(A.T, vec)
    idx = np.argsort(-val)
    val = val[idx]
    vec = vec[:, idx]
    n = A.shape[0]
    for i in range(n):
        vec[:,i] = vec[:,i]/np.linalg.norm(vec[:,i])
#     P = np.dot(vec.T, C.T)
#     return P.T
    return [val, vec, M]


# # creating the datasets

# In[20]:


def create_dataset(path_to_images):
    X = []
    y = []
    c=1
    for dirname, dirnames, filenames in os.walk(path_to_images):
#         print(filenames)
        for i in filenames:
            print(i)
            im = cv2.imread(os.path.join(path_to_images, i))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            X.append(im)
            y.append(c)
    return [X, y]


# # Normalize function

# In[10]:


def Normalize(X, high, low):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].	
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    return np.asarray(X)


# In[25]:


def asRowMatrix(X):
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
    return mat


# # project and normalize

# In[46]:


def project(W, X, mu=None):
    if mu is None:
        return np.dot(X,W)
    return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y,W.T)
    return np.dot(Y, W.T) + mu


# In[42]:


def create_font(fontname='Tahoma', fontsize=10):
    return { 'fontname': fontname, 'fontsize':fontsize }
def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center') 
    for i in range(len(images)):
        ax0 = fig.add_subplot(rows,cols,(i+1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


# ## # this is main

# In[30]:


X, y = create_dataset("images/haresh/")


# In[37]:


val, vec, mu = PCA(asRowMatrix(X), y)


# In[38]:


E = []


# In[41]:


for i in range(len(X)):
    e = vec[:,i].reshape(X[0].shape)
    E.append(Normalize(e,0,255))


# In[43]:


subplot(title="Eigenfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="eigenfaces.png")


# In[44]:


steps=[i for i in range(10, min(len(X), 320), 5)]


# In[49]:


E = []
for i in range(min(len(steps), 16)):
    numEvs = steps[i]
    P = project(vec[:,0:numEvs], X[0].reshape(1,-1), mu)
    R = reconstruct(vec[:,0:numEvs], P, mu)
    # reshape and append to plots
    R = R.reshape(X[0].shape)
    E.append(Normalize(R,0,255))
subplot(title="Reconstruction AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenvectors", sptitles=steps, colormap=cm.gray, filename="python_pca_reconstruction.png")


# In[50]:


def Euclidian(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sqrt(np.sum(np.power((p-q),2)))


# In[ ]:





# In[2]:


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detect(gray, 0)
    for rect in rects:
        shape = predict(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(img, (x,y), 2, (0,255,0), -1)
    cv2.imshow("output", img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

