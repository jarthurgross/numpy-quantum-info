import numpy as np

def Jm_mat(s):
    dim = int(2*s + 1)
    return np.array([[np.sqrt((s + 1)*(a + b + 1) - (a + 1)*(b + 1))
                      if a==b+1 else 0 for b in range(dim)]
                     for a in range(dim)], dtype=np.complex)

def Jx_mat(s):
    Jm = Jm_mat(s)
    return (np.conj(Jm).T + Jm)/2

def Jy_mat(s):
    Jm = Jm_mat(s)
    return (np.conj(Jm).T - Jm)/(2j)

def Jz_mat(s):
    dim = int(2*s + 1)
    return np.array([[s - a if a==b else 0 for b in range(dim)]
                     for a in range(dim)], dtype=np.complex)
