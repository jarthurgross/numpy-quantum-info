"""Tools for working with superoperators (linear operators on operator spaces).

"""

from functools import reduce

import numpy as np
import sparse
from sparse import COO

import qinfo as qi

###############################################################################
# Action routines
###############################################################################

def act_proc_mat(operator, proc_mat, op_basis):
    """Act a process matrix on an operator

    Parameters
    ----------
    operator: array_like
        The operator to be acted upon
    proc_mat: array_like
        The process matrix with respect to the given operator basis
    op_basis: OperatorBasis
        Basis of operators used to express the process matrix and operator
        vector

    Returns
    -------
    numpy.array
        The matrix form of the image of the vectorized operator under the
        process matrix

    """
    op_vec = op_basis.vectorize(operator)
    return op_basis.matrize_vector(proc_mat @ op_vec)

def act_MA_tensot(operator, ma_tensor, op_basis):
    """Calculate the middle action of a tensor on an operator.

    The middle action of a tensor A_jk on an operator B, with respect to an
    operator basis {X_j}, is the Kraus-operator way of acting with the tensor:
    B -> A_jk*X_j*B*X^dag_k.

    Parameters
    ----------
    operator : array_like
        The operator to be acted upon
    ma_tensor : array_like
        The middle-action tensor
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

def act_LR_tensor(operator, lr_tensor, op_basis):
    """Calculate the left-right action of the tensor on the argument.

    The left-right action of a tensor A_jk on an operator B, with respect to an
    operator basis {X_j}, is the operator bra-ket way of acting with the tensor:
    B -> A_jk*tr(B*X^dag_k)*X_j.

    Parameters
    ----------
    operator : array_like
        The operator to be acted upon
    lr_tensor : array_like
        The left-right-action tensor
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

def act_proc_tensor(operator, proc_tensor):
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
    operator : np.array
        The operator on which to act the process tensor
    proc_tensor : np.array
        The process tensor

    Returns
    -------
    numpy.array
        The image of the operator under the process

    """
    return np.einsum('jkmn,jk->mn', proc_tensor, operator)

###############################################################################
# Miscellaneous routines
###############################################################################

def proc_tensor_compose(t2, t1):
    '''Combine tensors to yield the tensor for the composed processes.

    Parameters
    ----------
    t1 : array_like
        Process tensor for the process preceeding the second process
    t2 : array_like
        Process tensor for the process following the first process

    Returns
    -------
    array_like
        The process tensor for the composition of the second process after the
        first process: T2 o T1

    '''
    return np.einsum('pqmn,jkpq->jkmn', t2, t1)

def compose_proc_tensors(proc_tensors):
    '''Compose many process tensors together.

    Processes follow processes that are further right in the list.

    '''
    return reduce(proc_tensor_compose, proc_tensors)


def kraus_eig(proc_tensor, mat_unit_basis=None):
    '''Perform an eigendecomposition of a process into Kraus operators.

    Uses the canonical Kraus decomposition where the Kraus operators are
    orthogonal to one another in Hilbert-Schmidt inner product. This routine
    returns a set of orthonormal operators {V_j} and corresponding eigenvalues
    {w_j} such that the Kraus operators are

    K_j = sqrt(w_j) V_j

    Parameters
    ----------
    proc_tensor : array_like
        The process tensor (a reshaping of the process matrix or equivalently
        left-right-action tensor in the matrix-unit basis)
    mat_unit_basis : MatrixUnitBasis
        Matrix-unit basis of the appropriate dimension (will be constructed if
        not provided).

    Returns
    -------
        Pair (w, V), where w is the array of eigenvalues and V is the list of
        orthonormal eigenoperators.

    '''
    if mat_unit_basis is None:
        dim_in = proc_tensor.shape[0]
        dim_out = proc_tensor.shape[2]
        mat_unit_basis = qi.MatrixUnitBasis(dim_in, dim_out)
    proc_mat = proc_tensor_to_mat_unit_proc_mat(proc_tensor)
    w, v = np.linalg.eigh(proc_mat)
    V = [mat_unit_basis.matrize_vector(v[:,n]) for n in range(w.shape[0])]
    return w, V

###############################################################################
# Conversion routines
###############################################################################

def process_to_proc_tensor(process, dim_in):
    """Calculate the process tensor given a process.

    The process tensor for a process E in a vector basis {|n>} has the following
    components:
    T_jkmn = <m| E(|j><k|) |n>

    The input and output Hilbert spaces may have different dimensions.

    Parameters
    ----------
    process : callable
        The process as a function that takes a density matrix and returns the
        resulting density matrix.
    dim_in : positive integer
        Dimension of the Hilbert space on which the input density operators act

    Returns
    -------
    numpy.array
        The process tensor

    """
    kets = [np.array([1 if j==k else 0 for k in range(dim_in)])
            for j in range(dim_in)]
    rho0_jks = [[np.outer(ket1, ket2.conj()) for ket2 in kets]
                for ket1 in kets]
    rho_mns = [[process(rho0) for rho0 in rho0_ks] for rho0_ks in rho0_jks]
    return np.array(rho_mns)

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

def proc_tensor_to_mat_unit_LR_tensor(proc_tensor):
    """Calculate the left-right-action tensor from the process tensor.

    The left-right-action tensor calculated is represented in the matrix-unit
    basis.

    Parameters
    ----------
    proc_tensor : array_like
        The process tensor

    Returns
    -------
        The left-right-action tensor

    """
    vec_dim = proc_tensor.shape[0]
    return proc_tensor.reshape(2*[vec_dim**2]).T

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
    return sparse.tensordot(lr_tensor, op_basis.get_gram_matrix(),
                            ([-1], [0]))

def proc_tensor_to_mat_unit_proc_mat(proc_tensor):
    """Permute and combine indices to make process matrix from process tensor.

    The process matrix is expressed in the matrix-unit basis consistent with the
    Hilbert-space basis in which the process tensor is defined.

    Parameters
    ----------
    proc_tensor : array_like
        The process tensor

    Returns
    -------
        The corresponding process matrix in the matrix-unit basis

    """
    dim_in = proc_tensor.shape[0]
    dim_out = proc_tensor.shape[2]
    # A_(mj)(nk) = T_jkmn
    return np.transpose(proc_tensor, (2, 0, 3, 1)).reshape(2*[dim_out*dim_in])

def mat_unit_proc_mat_to_proc_tensor(proc_mat, dim_in, dim_out):
    """Split and permute indices to make process tensor from process matrix.

    The process matrix is expressed in the matrix-unit basis consistent with the
    Hilbert-space basis in which the process tensor is defined.

    Parameters
    ----------
    proc_mat : array_like
        The process matrix in the matrix-unit basis
    dim_in : positive integer
        The dimension of the input Hilbert space
    dim_out : positive integer
        The dimension of the output Hilbert space

    Returns
    -------
        The corresponding process tensor

    """
    # T_jkmn = A_(mj)(nk)
    return np.transpose(proc_mat.reshape(2*[dim_out, dim_in]), (1, 3, 0, 2))

def proc_tensor_to_kraus_decomp(proc_tensor, mat_unit_basis=None):
    """Calculate the Kraus operator for a process given a process tensor.

    Parameters
    ----------
    proc_tensor : array_like
        The process tensor (a reshaping of the process matrix or equivalently
        left-right-action tensor in the matrix-unit basis)
    mat_unit_basis : MatrixUnitBasis
        Matrix-unit basis of the appropriate dimension (will be constructed if
        not provided).

    Returns
    -------
        List of operators representing the Kraus operators

    """
    w, V = kraus_eig(proc_tensor, mat_unit_basis)
    # We'll assume the process is actually a CPTP map and truncate the
    # eigenvalues at 0.
    return [np.sqrt(max(0, w[n])) * V[n] for n in range(w.shape[0])]

def kraus_decomp_to_proc_tensor(kraus_decomp):
    raise NotImplementedError()

def proc_tensor_to_choi_mat(proc_tensor):
    """Get the Choi matrix of the process from the process tensor.

    The Choi matrix is the image of half of an unnormalized maximally
    entangled state under the process:

    C_Phi = sum_jk E_jk o Phi(E_jk)

    where E_jk = |j><k|.

    See https://en.wikipedia.org/wiki/Choi%27s_theorem_on_completely_positive_maps

    This is easily obtained from the process tensor:

    C_Phi(jm)(kn) = T_jkmn

    Parameters
    ----------
    proc_tensor : np.array
        The process tensor

    Returns
    -------
    numpy.array
        The Choi matrix

    """
    s = proc_tensor.shape
    return np.transpose(proc_tensor, (0, 2, 1, 3)).reshape(s[0]*s[2], s[1]*s[3])

def choi_mat_to_proc_tensor(choi_mat, dim_in=None):
    """Get the process tensor from the Choi matrix of the process.

    The Choi matrix is the image of half of an unnormalized maximally
    entangled state under the process:

    C_Phi = sum_jk E_jk o Phi(E_jk)

    where E_jk = |j><k|.

    See https://en.wikipedia.org/wiki/Choi%27s_theorem_on_completely_positive_maps

    The process tensor is easily obtained from this matrix:

    T_jkmn = C_Phi(jm)(kn)

    Parameters
    ----------
    choi_mat : np.array
        The Choi matrix
    dim_in : positive integer
        The dimension of the input Hilbert space. If `None`, is assumed to be
        the same as the dimension of the output Hilbert space and then inferred
        from the dimensions of the Choi matrix.

    Returns
    -------
    numpy.array
        The Choi matrix

    """
    s = choi_mat.shape
    if dim_in is None:
        dim_in = int(np.round(np.sqrt(s[0])))
        dim_out = dim_in
    else:
        dim_out = s[0]//dim_in
    assert dim_in*dim_out == s[0]
    return np.transpose(choi_mat.reshape(dim_in, dim_out, dim_in, dim_out), (0, 2, 1, 3))
