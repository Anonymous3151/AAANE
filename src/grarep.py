import math
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize

class GraRep(object):

    def __init__(self, graph, Kstep):
        self.g = graph
        self.Kstep = Kstep
        # assert dim%Kstep == 0
        # self.dim = dim/Kstep
        self.train()

    def getAdjMat(self):
        graph = self.g.G
        node_size = self.g.node_size

        print(node_size)
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
            adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
        # ScaleSimMat
        return np.matrix(adj/np.sum(adj, axis=1))

    def GetProbTranMat(self, Ak):
        probTranMat = np.log(Ak/np.tile(
            np.sum(Ak, axis=0), (self.node_size, 1))) \
            - np.log(1.0/self.node_size)
        probTranMat[probTranMat < 0] = 0
        probTranMat[probTranMat == np.nan] = 0
        return probTranMat

    # def GetRepUseSVD(self, probTranMat, alpha):
    #     U, S, VT = la.svd(probTranMat)
    #     Ud = U[:, 0:self.dim]
    #     Sd = S[0:self.dim]
    #     return np.array(Ud)*np.power(Sd, alpha).reshape((self.dim))
    #
    # def save_embeddings(self, filename):
    #     fout = open(filename, 'w')
    #     node_num = len(self.vectors.keys())
    #     fout.write("{} {}\n".format(node_num, self.Kstep*self.dim))
    #     for node, vec in self.vectors.items():
    #         fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
    #     fout.close()

    def train(self):
        self.adj = self.getAdjMat()
        self.node_size = self.adj.shape[0]
        self.Ak = np.matrix(np.identity(self.node_size))
        self.RepMat = np.zeros((self.node_size, self.node_size*self.Kstep))
        for i in range(self.Kstep):
            print 'Kstep =', i
            self.Ak = np.dot(self.Ak, self.adj)
            probTranMat = self.GetProbTranMat(self.Ak)

            #print(probTranMat.shape)
            #Rk = self.GetRepUseSVD(probTranMat, 0.5)
            #Rk = normalize(Rk, axis=1, norm='l2')
            #print(Rk.shape)
            #self.RepMat[:, self.dim*i:self.dim*(i+1)] = Rk[:, :]

            Rk = normalize(probTranMat, axis=1, norm='l2')
            self.RepMat[:, self.node_size * i:self.node_size * (i + 1)] = Rk[:, :]

        print(self.RepMat.shape)

        #print(self.RepMat[:10,:10])

        #self.vectors = self.RepMat
        #get embeddings
        self.vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self.RepMat):
            self.vectors[look_back[i]] = embedding




