import numpy as np
import PIL.Image as Image
import cv2
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import dlib
import os

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

def create_dataset(path_to_images):
	X = []
	y = []
	images = []
	code = {}
	c=0
	for dirname, dirnames, filenames in os.walk(path_to_images):
		images.append(dirnames)
	#         print(filenames)
		print(images[0])
		for i in images[0]:
			for j in filenames:
				im = cv2.imread(os.path.join(os.path.join(path_to_images, i), j))
				im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				X.append(im)
				y.append(c)
				code[c] = i
			c+=1
	return [X, y, code]

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

def asRowMatrix(X):
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
    return mat

def project(W, X, mu=None):
    if mu is None:
        return np.dot(X,W)
    return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y,W.T)
    return np.dot(Y, W.T) + mu

def euclidianDist(p, q):
	p = np.asarray(p).flatten()
	q = np.asarray(q).flatten()
	return np.sqrt(np.sum(np.power((p-q),2)))
