"""Tools for working with superoperators (linear operators on operator spaces).

"""

import numpy as np
import sparse
from sparse import COO

import qinfo as qi

def proc_mat_to_LR_tensor(proc_mat, op_basis):
    """Turn a process matrix into the corresponding left-right-action tensor.

    A process matrix A acts on vectorized operators via matrix-vector
    multiplication:

    b_j*X_j -> A_jk*b_k*X_j.

    A left-right-action tensor C acts on operators by multiplying the basis
    operator corresponding to the left index by the Hilbert-Schmidt inner
    product of the operator argument with the basis operator corresponding to
    the right index:

    B -> C_jk*tr(X^dag_k*B)*X_j.

    Parameters
    ----------
    proc_mat : array_like
        The process matrix with respect to the given operator basis
    op_basis : OperatorBasis
        Basis of operators used to express the process matrix

    Returns
    -------
    numpy.array
        The tensor whose left-right action is equivalent to the matrix-vector
        action of the process matrix in the given basis

    """

    # rho = r_j*X_j
    # E(rho) = A_jk*r_k*X_j
    # (X_j|E(X_k)) = A_jk*(X_j|X_j)
    # A_jk = (X_j|E(X_k))/(X_j|X_j)
    # E(rho) = E_jk*(X_k|rho)*X_j
    #        = E_jk*r_m*(X_k|X_m)*X_j
    #        = E_jm*(X_m|X_k)*r_k*X_j
    # E_jm*IP_mk = A_jk
    # E_jk = A_jm*IP.inv()_mk
    return sparse.tensordot(proc_mat, op_basis.get_gram_matrix_inv(),
                            ([-1], [0]))

def LR_tensor_to_proc_mat(lr_tensor, op_basis):
    """Turn a left-right-action tensor into the corresponding process matrix.

    A process matrix A acts on vectorized operators via matrix-vector
    multiplication:

    b_j*X_j -> A_jk*b_k*X_j.

    A left-right-action tensor C acts on operators by multiplying the basis
    operator corresponding to the left index by the Hilbert-Schmidt inner
    product of the operator argument with the basis operator corresponding to
    the right index:

    B -> C_jk*tr(X^dag_k*B)*X_j.

    Parameters
    ----------
    lr_tensor : array_like
        The left-right-action tensor with respect to the given operator basis
    op_basis : OperatorBasis
        Basis of operators used to express the process matrix

    Returns
    -------
    numpy.array
        The process matrix whose action is equivalent to the left-right
        action of the left-right-action tensor in the given basis

    """

    # rho = r_j*X_j
    # E(rho) = A_jk*r_k*X_j
    # (X_j|E(X_k)) = A_jk*(X_j|X_j)
    # A_jk = (X_j|E(X_k))/(X_j|X_j)
    # E(rho) = E_jk*(X_k|rho)*X_j
    #        = E_jk*r_m*(X_k|X_m)*X_j
    #        = E_jm*(X_m|X_k)*r_k*X_j
    # E_jm*IP_mk = A_jk
    # E_jk = A_jm*IP.inv()_mk
    return sparse.tensordot(proc_mat, op_basis.get_gram_matrix(),
                            ([-1], [0]))

def proc_action(proc_mat, op_vec, op_basis):
    """Get the matrix form of a process matrix times a vectorized operator.

    Parameters
    ----------
    proc_mat: array_like
        The process matrix with respect to the given operator basis
    op_vec: array_like
        The vectorized operator with respect to the given operator basis
    op_basis: OperatorBasis
        Basis of operators used to express the process matrix and operator
        vector

    Returns
    -------
    numpy.array
        The matrix form of the image of the vectorized operator under the
        process matrix

    """
    return op_basis.matrize_vector(proc_mat @ op_vec)

def middle_action(ma_tensor, operator, op_basis):
    """Calculate the middle action of a tensor on an operator.

    The middle action of a tensor A_jk on an operator B, with respect to an
    operator basis {X_j}, is the Kraus-operator way of acting with the tensor:
    B -> A_jk*X_j*B*X^dag_k.

    Parameters
    ----------
    ma_tensor : array_like
        The middle-action tensor
    operator : array_like
        The operator to be acted upon
    op_basis : OperatorBasis
        Basis of operators expressed as matrices

    Returns
    -------
    numpy.array
        The operator resulting from the action

    """
    if isinstance(ma_tensor, np.ndarray):
        ma_tensor = COO.from_numpy(ma_tensor)
    if isinstance(operator, np.ndarray):
        operator = COO.from_numpy(operator)
    return sparse.tensordot(sparse.tensordot(ma_tensor,
                                             op_basis.get_operators(),
                                             ([-2], [0])),
                            sparse.tensordot(operator,
                                             op_basis.get_operators().conj(),
                                             ([-1], [2])),
                            ([-3, -1], [-2, -3]))

def left_right_action(lr_tensor, operator, op_basis):
    """Calculate the left-right action of the tensor on the argument.

    The left-right action of a tensor A_jk on an operator B, with respect to an
    operator basis {X_j}, is the operator bra-ket way of acting with the tensor:
    B -> A_jk*tr(B*X^dag_k)*X_j.

    Parameters
    ----------
    ma_tensor : array_like
        The left-right-action tensor
    operator : array_like
        The operator to be acted upon
    op_basis : OperatorBasis
        Basis of operators expressed as matrices

    Returns
    -------
    numpy.array
        The operator resulting from the action

    """
    if isinstance(lr_tensor, np.ndarray):
        lr_tensor = COO.from_numpy(lr_tensor)
    if isinstance(operator, np.ndarray):
        operator = COO.from_numpy(operator)
    return sparse.tensordot(sparse.tensordot(lr_tensor,
                                             op_basis.get_operators(),
                                             ([-2], [0])),
                            sparse.tensordot(operator,
                                             op_basis.get_operators().conj(),
                                             ([-2, -1], [1, 2])),
                            ([-3], [-1]))

def get_process_tensor_from_process(process, dim):
    """Calculate the process tensor given a process.

    The process tensor for a process E in a vector basis {|n>} has the following
    components:
    T_jkmn = <m| E(|j><k|) |n>

    Parameters
    ----------
    process : callable
        The process as a function that takes a density matrix and returns the
        resulting density matrix.
    dim : int
        Dimension of the Hilbert space on which the density operators act

    Returns
    -------
    numpy.array
        The process tensor

    """
    kets = [np.array([1 if j==k else 0 for k in range(dim)])
            for j in range(dim)]
    rho0_jks = [[np.outer(ket1, ket2.conj()) for ket2 in kets]
                for ket1 in kets]
    rho_jks = [[process(rho0) for rho0 in rho0_ks] for rho0_ks in rho0_jks]
    return np.array(rho_jks)

def act_process_tensor(proc_tensor, operator):
    """Calculate the action of a process tensor on a state.

    The process tensor for a process E in a vector basis {|n>} has the following
    components:
    T_jkmn = <m| E(|j><k|) |n>
    For a state rho = rho_jk |j><k| the action of the process is calculated as
    below:
    E(rho) = T_jkmn rho_jk |m><n|
    so the new density matric elements are given by this particular contraction
    of T with the original density matrix elements.

    Note that the process tensor is a reshaped version of the process matrix and
    the left-right-action tensor for the process in the matrix-unit basis.

    Parameters
    ----------
    proc_tensor : np.array
        The process tensor
    operator : np.array
        The operator on which to act the process tensor

    Returns
    -------
    numpy.array
        The image of the operator under the process

    """
    return np.einsum('jkmn,jk->mn', proc_tensor, operator)

def get_process_state_from_tensor(proc_tensor):
    """Get the Choi state of the process from the process tensor.

    The Choi state is a partial transpose of the image of half of a maximally
    entangled state under the process. This is easily obtained from the process
    tensor: chi_(mk,nj) = (1 / d_in) T_jkmn

    Parameters
    ----------
    proc_tensor : np.array
        The process tensor

    Returns
    -------
    numpy.array
        The Choi state

    """
    s = proc_tensor.shape
    dim_in = s[0]
    return (1/dim_in)*np.transpose(proc_tensor,
                                   (2, 0, 3, 1)).reshape((s[2]*s[0],
                                                          s[3]*s[1]))
