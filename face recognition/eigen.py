from util import PCA, asRowMatrix, project,  euclidianDist
import numpy as np
class EigenFaces(object):
	"""docstring for EigenFaces"""
	def __init__(self, X=None, y=None):
		self.X = X
		self.y = y
		# self.dist_metric = dist_metric
		# self.num_components = num_components
		self.W = []
		self.mu = []
		self.projections = []
	def train(self, X, y):
		[D, self.W, self.mu] = PCA(asRowMatrix(X),y)
		self.y = y
		for xi in self.X:
			self.projections.append(project(self.W, xi.reshape(1,-1), self.mu))
	def predict(self, X):
		minDist = np.finfo('float').max
		minClass = -1
		Q = project(self.W, X.reshape(1,-1), self.mu)
		for i in range(len(self.projections)):
			dist = euclidianDist(self.projections[i], Q)
			if dist < minDist:
				minDist = dist
				minClass = self.y[i]
		# print("aaaaaaaaaaa")
		return minClass