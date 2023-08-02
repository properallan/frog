import numpy as np
import h5py
import matplotlib.pyplot as plt

class SVD:
    def __init__(self, snapshots):
        if type(snapshots) == type('string'):
            ds = h5py.File(snapshots,'r')
            self.snapshots = ds['snapshots'][:]
            ds.close()
        else:
            self.snapshots = snapshots.T

        #length = int(self.snapshots[:].shape[0]/3)
        #self.snapshots = self.snapshots[int(2*length):int(3*length),:]
            
        self.normalized = False
        self.meanSubtracted = False

    def SVD(self, fSVD=np.linalg.svd, rank=None, **kwargs):
        self.fSVD = fSVD
        if self.fSVD == np.linalg.svd:
            self.uf, s, vT = self.fSVD(self.snapshots, full_matrices=False, **kwargs)
            self.vf = vT.T
            self.sf = np.diag(s)
 
        if rank is not None:
            self.setRank(rank)
        else:
            self.rank = len(self.sf.diagonal())
            self.u = self.uf
            self.s = self.sf
            self.v = self.vf
 
        return self.u, self.s, self.v

    def setRank(self, rank=None):
        if rank is not None:
            self.rank = rank

            self.u = self.uf[:,:rank]
            self.s = self.sf[:rank, :rank]
            self.v = self.vf[:, :rank]

        self.compress()

        return self.u, self.s, self.v

    def energy(self):
        s = self.s.diagonal()
        sf = self.sf.diagonal()
        return s/np.sum(sf)*100

    def energyFull(self):
        s = self.sf.diagonal()
        sf = self.sf.diagonal()
        return s/np.sum(sf)*100

    def plotEnergy(self, **kwargs):
        e = self.energy()
        if self.rank < len(self.sf.diagonal()):
            self.plotEnergyFull(lw=0, marker='o')
        plt.plot(np.arange(1,len(e)+1), e, **kwargs)
        
    def plotEnergyFull(self, **kwargs):
        e = self.energyFull()
        plt.plot(np.arange(1,len(e)+1), e, **kwargs)

    def acumEnergy(self):
        return np.cumsum(self.energy())[-1]

    def findRank(self, acumEnergyTarget):
        rank = 0
        actualAcumEnergy = 0
        while(acumEnergyTarget > actualAcumEnergy):
            rank += 1
            self.setRank(rank)
            actualAcumEnergy = self.acumEnergy()
        print('{:.2f} % of energy preserved at rank {}'.format(actualAcumEnergy, rank))
        return rank

    def compress(self, A=None):
        # coefficient matrix
        if A is None:
            #compute for all snapshots
            A = self.snapshots
            #self.L = (self.u.T @ A).T
            #L = self.L
        elif type(A) is type(0):
            # if A is an integer compute for the i-th snapthos
            A = self.snapshots[:,A]
            A = np.expand_dims(A, axis=1)
            #self.L = (self.u.T @ A).T
            #L = self.L
        # otherwhise the snapshot A should be given
            

        self.L = (self.u.T @ A).T

        '''if self.meanSubtracted:
            self.L = self.subtractMean(self.L)   
        if self.normalized:
            self.L = self.normalize(self.L, self.bounds)'''

        return self.L

    def reconstruct(self, tildeL=None):
        try:
            self.L
        except:
            print('cannot reconstruct an uncompressed data')
            return 1
        if tildeL is None:
            # reconstruct all snapshots
            tildeL = self.L
        if type(tildeL) is type(0):
            # if tildeL is an integer compute for the i-th coefficient column
            tildeL = self.L[tildeL,:]
            tildeL = np.expand_dims(tildeL, axis=0)

        # otherwhise the coeficcient matrix L should be given
        '''if self.meanSubtracted:
            tildeL = self.addMean(tildeL)   
        if self.normalized:
            tildeL = self.denormalize(tildeL)'''
        self.A = (self.u @ tildeL.T).T

        if self.meanSubtracted:
            self.A = self.addMean(self.A)   
        if self.normalized:
            self.A = self.denormalize(self.A)
            
        return self.A

    def subtractMean(self, X=None, mean = None):
        if X is None:
            X = self.snapshots
        if mean is None:
            if self.meanSubtracted:
                mean = self.mean
            else:
                mean = np.mean(X, axis=1)

        for i, col in enumerate(X[0,:]):
            X[:,i] = X[:,i] - mean

        if X.all() == self.snapshots.all():
            self.snapshots = X
            self.mean = mean
            self.meanSubtracted = True

        return X, mean

    def addMean(self, X=None):
        if X is None:
            X = self.snapshots
        mean = self.mean
        for i, col in enumerate(X[0,:]):
            X[:,i] = X[:,i] + mean
        if X is None:
            self.snapshots = X
        return X

    def normalize(self, data=None, bounds=None, min=None, max=None):
        if data is None:
            self.bounds = bounds
            data = self.snapshots
        if min is None and max is None:
            min = np.min(data)
            max = np.max(data)
        if self.normalized:
            bounds = self.bounds
            min = self.min
            max = self.max
            
        normdata = (bounds[1]-bounds[0])*(data - min)/(max-min) + bounds[0]
        
        if data.all() == self.snapshots.all():
            self.snapshots = normdata
            self.min = min
            self.max = max
            self.normalized = True

        return normdata, min, max

    def denormalize(self, data=None):
        if data is None:
            data = self.snapshots
        bounds = self.bounds
        min = self.min
        max = self.max
        denormdata = (data - bounds[0])*(max-min)/(bounds[1]-bounds[0]) + min
        
        if data is None:
            self.snapshots = denormdata

        return denormdata

    def getSnapshot(self, idx):
        snap = np.copy(self.snapshots[:,idx:idx+1])
        if self.meanSubtracted:
            snap = self.addMean(snap)   
        if self.normalized:
            snap = self.denormalize(snap)
        return snap