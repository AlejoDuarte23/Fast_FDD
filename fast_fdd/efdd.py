import numpy as np 

from numpy.typing import NDArray
from scipy import signal

class EFDD():
    def __init__(self, Acc: NDArray, fs: int, Nc: int) -> None:
        self.Acc = Acc
        self.fs = fs
        self.Nc = Nc
        self.N = Acc.shape[0]

    def get_psd_matrix(self):
        Nc = self.Nc
        AN = int(len(self.Acc[:,0])/2)+1 # nfft/2+1
        # Memory alocation for the matrix
        PSD = np.zeros((Nc,Nc,AN),dtype=np.complex_)
        freq= np.zeros((Nc,Nc,AN),dtype=np.complex_)

        for i in range(Nc):
            for j in range(Nc):
                f, Pxy = signal.csd(self.Acc[:,i], self.Acc[:,j], self.fs, nfft=AN*2-1,nperseg=2**11,noverlap = None,window='hamming')
                freq[i,j]= f
                PSD[i,j]= Pxy
        return PSD,freq
           
    def get_eigen_values(self):
        PSD,f = self.get_psd_matrix()
        #eigen values descomposition 
        s1 = np.zeros(len(f))
        for  i in range(len(f)):
            u, s, vh = np.linalg.svd(PSD[:,:,i], full_matrices=True)
            s1[i] = s[0]
        return s1
    

    def MacVal(self, Mode1,Mode2):
        ter1 =  Mode1.conj().transpose()
        ter2 =  Mode2.conj()
        num  = np.matmul(ter1, ter2)**2#np.matmul(ter1,ter2).shape
        den1  = np.matmul( Mode1.conj().transpose(), Mode1.conj())
        den2  = np.matmul( Mode2.conj().transpose(), Mode2.conj())
        den = np.matmul(den1,den2)
        Mac = num/den
        return Mac
