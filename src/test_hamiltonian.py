import quimb as qu
from classiq import *
import numpy as np
import scipy as sp

N = 4
J = 1

def revert_matrix(N):
    row = np.arange(2**N)
    col = np.zeros(2**N)
    data = np.ones(2**N)
    for i in range(2**N):
        bin_str = f'{{:0{N}b}}'.format(i)
        bin_str = bin_str[::-1]
        col[i] = int(bin_str, 2)
    mat = sp.sparse.csc_matrix((data, (row, col)), shape=(2**N, 2**N))
    return mat

def basis_creation(N):
    basis = []
    check_half = lambda s: sum([int(c) for c in s])
    for index in range(2**N):
        bin_str = f'{{:0{N}b}}'.format(index)
        tmp = check_half(bin_str)
        if tmp == N // 2:
            basis.append(qu.computational_state(bin_str, sparse=True))
    print(np.shape(basis))
    return sp.sparse.hstack(basis)

def basis_transform(obj, basis):
    if qu.isvec(obj):
        return basis @ obj
    else:
        return qu.dag(basis) @ obj @ basis
    
def get_eigstates(N):
    basis = basis_creation(N)
    ham = qu.ham_heis(N, j=1, cyclic=False)
    ham_reduce = basis_transform(ham, basis)
    eigstate_reduce = qu.eigvecsh(ham_reduce)
    # print(4 * qu.eigvalsh(ham_reduce))
    # print(4 * qu.eigvalsh(ham))
    eigstates = []
    for index in range(np.shape(eigstate_reduce)[1]):
        obj = qu.qu(eigstate_reduce[:, index], qtype='ket')
        obj = basis_transform(obj, basis)
        eigstates.append(obj)
    return eigstates
    
ham_matrix = qu.ham_heis(N, J, cyclic=False)
ham_matrix2 = qu.ham_XXZ(N, J, J,cyclic=False)
print(ham_matrix)
print(ham_matrix2)

rev_mat = revert_matrix(N)
print(qu.eigvalsh(ham_matrix) * 4)
eigenstates = get_eigstates(N)


target_state = eigenstates[0]
print(target_state)