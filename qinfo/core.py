"""Foundational components for working with vector spaces of operators.

"""

import itertools as it
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as spsp
import sparse
from sparse import COO

class OperatorBasis:
    """Object storing the matrices in an operator basis and additional details.

    The matrix representations of the operators are assumed to be expressed in
    an orthonormal basis for the underlying Hilbert space.

    Parameters
    ----------
    operators : array_like
        Array of shape (op_basis_size, vec_basis_size, vec_basis_size) storing
        matrix representations of all the operators in an operator basis for
        operators on a given vector space. Supported types are `sparse.COO` and
        `numpy.ndarray`.

    """
    def __init__(self, operators, **kwargs):
        self.__dict__ = kwargs
        if isinstance(operators, COO):
            self.operators = operators
        elif isinstance(operators, np.ndarray):
            self.operators = COO.from_numpy(operators)
        else:
            raise ValueError("Expected 'operators' to be of type 'sparse.COO' "
                             "or 'numpy.ndarray' but got object of type '{}'."
                             .format(type(operators)))
        self.op_dim = self.operators.shape[0]
        self.vec_dim = self.operators.shape[1]

    def get_operators(self):
        return self.operators

    def compute_gram_matrix(self):
        """Compute the matrix of Hilbert-Schmidt operator inner products.

        """
        self.gram_matrix = sparse.tensordot(self.operators,
                                            self.operators.conj(),
                                            ([1, 2], [1, 2])).tocsc()

    def set_gram_matrix(self, gram_matrix):
        """Set the Gram matrix for this operator basis.

        Parameters
        ----------
        gram_matrix : array_like
            The Gram matrix for this operator basis.

        """
        if isinstance(gram_matrix, COO):
            self.gram_matrix = gram_matrix.tocsc()
        elif isinstance(gram_matrix, np.ndarray):
            self.gram_matrix = spsp.csc_matrix(gram_matrix)
        else:
            self.gram_matrix = gram_matrix

    def get_gram_matrix(self):
        """Get the Gram matrix for this operator basis.

        """
        if not hasattr(self, 'gram_matrix'):
            self.compute_gram_matrix()
        return self.gram_matrix

    def compute_gram_matrix_inv(self):
        """Store the inverse of the Gram matrix for this operator basis.

        """
        if not hasattr(self, 'gram_matrix'):
            self.compute_gram_matrix()
        self.gram_matrix_inv = spla.inv(self.gram_matrix)

    def set_gram_matrix_inv(self, gram_matrix_inv):
        """Set the inverse of the Gram matrix for this operator basis.

        Parameters
        ----------
        gram_matrix_inv : array_like
            The inverse of the Gram matrix for this operator basis.

        """
        if isinstance(gram_matrix_inv, COO):
            self.gram_matrix_inv = gram_matrix_inv.tocsc()
        elif isinstance(gram_matrix_inv, np.ndarray):
            self.gram_matrix_inv = spsp.csc_matrix(gram_matrix_inv)
        else:
            self.gram_matrix_inv = gram_matrix_inv

    def get_gram_matrix_inv(self):
        """Get the inverse of the Gram matrix for this operator basis.

        """
        if not hasattr(self, 'gram_matrix_inv'):
            self.compute_gram_matrix_inv()
        return self.gram_matrix_inv

    def compute_dual_operators(self):
        """Store the dual operator basis.

        Hilbert-Schmidt inner products between the dual operator basis and the
        operator basis make kronecker deltas.

        """
        if not hasattr(self, 'gram_matrix_inv'):
            self.compute_gram_matrix_inv()
        self.dual_operators = sparse.tensordot(
                COO.from_scipy_sparse(self.gram_matrix_inv),
                self.operators, ([1], [0]))

    def set_dual_operators(self, dual_operators):
        """Set the dual operator basis.

        Parameters
        ----------
        dual_operators : array_like
            The dual operator basis.

        """
        if isinstance(dual_operators, np.ndarray):
            self.dual_operators = COO.from_numpy(dual_operators)
        else:
            self.dual_operators = dual_operators

    def vectorize(self, operator):
        """Calculate the components for an operator in this basis.

        `OperatorBasis.vectorize` and `OperatorBasis.matrize_vector` are
        inverses of one another.

        Parameters
        ----------
        operator : array_like
            The operator to vectorize

        """
        if not hasattr(self, 'dual_operators'):
            self.compute_dual_operators()
        if isinstance(operator, np.ndarray):
            operator = COO.from_numpy(operator)
        return sparse.tensordot(operator, self.dual_operators.conj(),
                                ([-2, -1], [1, 2]))

    def dualize(self, operator):
        """Calculate the components for an operator functional.

        Returns a vector such that the dot product of this vector with the
        vectorization of another operator is equal to the Hilbert-Schmidt inner
        product between this operator and the other operator, with this
        operator in the antilinear position.

        `OperatorBasis.dualize` and `OperatorBasis.matrize_dual` are
        inverses of one another.

        Parameters
        ----------
        operator : array_like
            The operator to dualize

        """
        if isinstance(operator, np.ndarray):
            operator = COO.from_numpy(operator)
        return sparse.tensordot(operator.conj(), self.operators,
                                ([-2, -1], [1, 2]))

    def matrize_vector(self, vector):
        """Calculate the matrix representation of a vectorized operator.

        `OperatorBasis.vectorize` and `OperatorBasis.matrize_vector` are
        inverses of one another.

        Parameters
        ----------
        vector : array_like
            The vectorized operator to express as a matrix.

        Returns
        -------
            The matrix representation of the vectorized operator.

        """
        if isinstance(vector, np.ndarray):
            vector = COO.from_numpy(vector)
        return sparse.tensordot(vector, self.operators, ([-1], [0]))

    def matrize_dual(self, dual):
        """Calculate the matrix representation of a dualized operator.

        `OperatorBasis.dualize` and `OperatorBasis.matrize_dual` are
        inverses of one another.

        Parameters
        ----------
        dual : array_like
            The dualized operator to express as a matrix.

        Returns
        -------
            The matrix representation of the dualized operator.

        """
        if not hasattr(self, 'dual_operators'):
            self.compute_dual_operators()
        if isinstance(dual, np.ndarray):
            dual = COO.from_numpy(dual)
        return sparse.tensordot(dual.conj(), self.dual_operators, ([-1], [0]))

    def compute_sharp_op(self):
        """Compute the tensor used to take the sharp of a superoperator tensor.

        """
        # Tensor product of the basis with itself
        # B_mnpq_jk = X_j_mn*X_k_qp
        B = sparse.tensordot(self.operators, self.operators.conj(),
                             0).transpose([1, 2, 5, 4, 0, 3])
        B_inv = COO.from_scipy_sparse(spla.inv(B.reshape(
            [self.op_dim**2, self.op_dim**2]).tocsc())).reshape(
                    2*[self.op_dim] + 4*[self.vec_dim])
        self.sharp_op = sparse.tensordot(B_inv, B, ([2,3,4,5], [0,3,2,1]))

    def set_sharp_op(self, sharp_op):
        """Set the tensor used to take the sharp of a superoperator tensor. 

        Parameters
        ----------
        sharp_op : array_like
            The tensor used to take the sharp of a superoperator tensor. 

        """
        if isinstance(sharp_op, np.ndarray):
            self.sharp_op = COO.from_numpy(sharp_op)
        else:
            self.sharp_op = sharp_op

    def sharp_tensor(self, tensor):
        """Perform the sharp on the tensor with respect to an operator basis.

        The sharp is an involution that swaps the middle and left-right actions
        of a tensor.  The is, the left-right action of A is equal to the middle
        action of A#, and vice versa.

        Parameters
        ----------
        tensor : numpy.array
            The tensor to sharp

        Returns
        -------
        array_like
            The sharp of the supplied tensor with respect to the given operator
            basis

        """
        if not hasattr(self, 'sharp_op'):
            self.compute_sharp_op()
        if isinstance(tensor, np.ndarray):
            tensor = COO.from_numpy(tensor)
        return sparse.tensordot(tensor, self.sharp_op, ([-2, -1], [2, 3]))

class OrthonormalOperatorBasis(OperatorBasis):
    """An `OperatorBasis` where the operators are guaranteed to be orthonormal.

    """
    def compute_gram_matrix(self):
        """Compute the matrix of Hilbert-Schmidt operator inner products.

        """
        self.gram_matrix = spsp.eye(self.op_dim, format='csc')

    def compute_gram_matrix_inv(self):
        """Store the inverse of the Gram matrix for this operator basis.

        """
        self.gram_matrix_inv = spsp.eye(self.op_dim, format='csc')

    def compute_dual_operators(self):
        """Store the dual operator basis.

        Hilbert-Schmidt inner products between the dual operator basis and the
        operator basis make kronecker deltas.

        """
        self.dual_operators = self.operators

class MatrixUnitBasis(OrthonormalOperatorBasis):
    """An `OperatorBasis` consisting of matrix units.

    Matrix units are matrices where all entries are 0 except for a single 1, and
    they form an orthonormal nonhermitian operator basis.

    Parameters
    ----------
    vec_dim : integer
        The dimension of the Hilbert space on which the operators act

    """
    def __init__(self, vec_dim_in, vec_dim_out=None):
        if vec_dim_out is None:
            vec_dim_out = vec_dim_in
        coords = np.array([[n, j, k] for n, (j, k)
                           in enumerate(it.product(range(vec_dim_out),
                                                   range(vec_dim_in)))]).T
        data = np.ones(vec_dim_in*vec_dim_out, dtype=np.complex)
        operators = COO(coords, data)
        super().__init__(operators)

    def compute_sharp_op(self):
        """Compute the tensor used to take the sharp of a superoperator tensor.

        For the matrix-unit basis, the `B` operator is orthogonal, so it's
        inverse is simply a transpose.

        """
        # Tensor product of the basis with itself
        # B_mnpq_jk = X_j_mn*X_k_qp
        B = sparse.tensordot(self.operators, self.operators.conj(),
                             0).transpose([1, 2, 5, 4, 0, 3])
        B_inv = B.reshape([self.op_dim**2,
            self.op_dim**2]).transpose().reshape(2*[self.op_dim] +
                    4*[self.vec_dim])
        self.sharp_op = sparse.tensordot(B_inv, B, ([2,3,4,5], [0,3,2,1]))
