import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans

# GMM Class
class GMM():
    def __init__(self, k, initialization='kmean'):
        self.k = k
        self.initialization = initialization
        self.data = None

    def _register_data(self, data):
        self.data = data
        self.n, self.d = data.shape
        
        # centers/means init
        if self.initialization == 'kmean':
            kmeans = KMeans(n_clusters=self.k, random_state=0, max_iter=1, tol=1, algorithm='elkan').fit(self.data)
            self.mean = np.array(kmeans.cluster_centers_)
        else:
            idx = np.random.randint(self.n, size=self.k)
            self.mean = np.array(self.data[idx,:])
        self.mean_trace = np.expand_dims(self.mean, -1)
        
        # affectation
        self.cluster = np.zeros(self.n)

        # responsabilities
        self.resp = np.zeros((self.n, self.k))
        self.N = self.resp.sum()
        
        # covariances
        self.cov = np.array([np.identity(self.d)]*self.k)

        # mixture weights
        self.pi = [1/self.k]*self.k

    def e_step(self):
        pdf = np.zeros((self.n, self.k))
        for j in range(self.k):
            pdf[:, j] = mvn.pdf(self.data, mean=self.mean[j], cov=self.cov[j])
        
        tot = pdf.sum(1)
        for j in range(self.k):
            self.resp[:, j] = self.pi[j] * pdf[:, j] / tot
    
        self.N = self.resp.sum()
        self.cluster = np.argmax(self.resp, axis=1)

    def m_step(self):
        nk = self.resp.sum(0)
        for j in range(self.k):
            nj = nk[j]

            # update mean
            muj = 1 / nj * self.resp[:, j].T.dot(self.data)
            self.mean[j] = muj

            # update cov
            centered = self.data - self.mean[j]
            outer = np.expand_dims(centered, -1) * np.expand_dims(centered, -2)
            wouter = self.resp[:, j].reshape(-1,1,1) * outer
            sigmaj = wouter.sum(0) / nj
            
            self.cov[j] = sigmaj
            self.pi[j] = nj / self.N

    def fit(self, data, iter=10):
        if self.data is None:
            self._register_data(data)
            
        for i in range(iter):
            self.e_step()
            self.m_step()
            self.mean_trace = np.concatenate((self.mean_trace, np.expand_dims(self.mean, -1)), 2)
        self.e_step()
        return self
