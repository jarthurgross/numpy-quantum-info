"""Foundational components for working with vector spaces of operators.

"""

import numpy as np
import scipy.sparse.linalg as spla
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

    def compute_gram_matrix(self):
        """Compute the matrix of Hilbert-Schmidt operator inner products.

        """
        self.gram_matrix = sparse.tensordot(self.operators,
                                            self.operators.conj(),
                                            ([1, 2], [1, 2])).tocsc()

    def compute_gram_matrix_inv(self):
        """Store the inverse of the Gram matrix for this operator basis.

        """
        if not hasattr(self, 'gram_matrix'):
            self.compute_gram_matrix()
        self.gram_matrix_inv = spla.inv(self.gram_matrix)

    def compute_dual_operators(self):
        if not hasattr(self, 'gram_matrix_inv'):
            self.compute_gram_matrix_inv()
        self.dual_operators = sparse.tensordot(
                COO.from_scipy_sparse(self.gram_matrix_inv),
                self.operators, ([1], [0]))

    def vectorize(self, operator):
        """Calculate the components for an operator in this basis.

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

        Parameters
        ----------
        operator : array_like

        """
        if isinstance(operator, np.ndarray):
            operator = COO.from_numpy(operator)
        return sparse.tensordot(operator.conj(), self.operators,
                                ([-2, -1], [1, 2]))

    def matrize_vector(self, vector):
        if isinstance(vector, np.ndarray):
            vector = COO.from_numpy(vector)
        return sparse.tensordot(vector, self.operators, ([-1], [0]))

    def matrize_dual(self, dual):
        if not hasattr(self, 'dual_operators'):
            self.compute_dual_operators()
        if isinstance(dual, np.ndarray):
            dual = COO.from_numpy(dual)
        return sparse.tensordot(dual.conj(), self.dual_operators, ([-1], [0]))
