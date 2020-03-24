from nose.tools import assert_almost_equal, assert_equal, assert_true
from scipy.linalg import sqrtm
from qinfo import supops

import numpy as np

def check_mat_eq(A, B):
    assert_almost_equal(np.linalg.norm(A - B), 0, 7)

def test_proc_tensor_state():
    RS = np.random.RandomState()
    RS.seed(2003231954)
    rand_tensor = RS.standard_normal((2, 2, 3, 3)) + 1.j*RS.standard_normal((2, 2, 3, 3))
    check_mat_eq(rand_tensor,
                 supops.choi_mat_to_proc_tensor(supops.proc_tensor_to_choi_mat(rand_tensor),
                                                dim_in=2))
    rand_mat = RS.standard_normal((6, 6)) + 1.j*RS.standard_normal((6, 6))
    check_mat_eq(rand_mat,
                 supops.proc_tensor_to_choi_mat(supops.choi_mat_to_proc_tensor(rand_mat,
                                                                               dim_in=2)))

def test_kraus():
    Id = np.eye(2, dtype=np.complex)
    sigx = np.array([[0, 1], [1, 0]], dtype=np.complex)
    sigy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
    sigz = np.array([[1, 0], [0, -1]], dtype=np.complex)

    def dephase_process(rho):
        return (rho + sigz @ rho @ sigz)/2
    def depol_process(rho):
        return (rho + sum([X @ rho @ X for X in [sigx, sigy, sigz]]))/4

    RS = np.random.RandomState()
    RS.seed(20031529)

    qutrit_unnorm_Ks = [RS.standard_normal((3, 3))
                        + 1j*RS.standard_normal((3, 3)) for j in range(8)]
    overcompleteness = sum([np.conj(K).T @ K for K in qutrit_unnorm_Ks])
    overcomp_factor = np.max(np.linalg.eigvalsh(overcompleteness))
    qutrit_Ks = ([K/np.sqrt(overcomp_factor) for K in qutrit_unnorm_Ks]
                 + [sqrtm(np.eye(3, dtype=np.complex)
                          - overcompleteness/overcomp_factor)])

    unnorm_2_to_3_Ks = [RS.standard_normal((3, 2))
                        + 1j*RS.standard_normal((3, 2)) for j in range(3)]
    overcomp_2_to_3 = sum([np.conj(K).T @ K for K in unnorm_2_to_3_Ks])
    overcomp_2_to_3_fac = np.max(np.linalg.eigvalsh(overcomp_2_to_3))
    Ks_2_to_3 = ([K/np.sqrt(overcomp_2_to_3_fac) for K in unnorm_2_to_3_Ks]
                 + [np.vstack([sqrtm(np.eye(2, dtype=np.complex)
                                     - overcomp_2_to_3/overcomp_2_to_3_fac),
                               np.zeros(2)])])
    def act_kraus_ops(rho, Ks):
        return sum([K @ rho @ np.conj(K).T for K in Ks])
    def random_qutrit_process(rho):
        return act_kraus_ops(rho, qutrit_Ks)
    def process_2_to_3(rho):
        return act_kraus_ops(rho, Ks_2_to_3)

    random_qubit_op = (RS.standard_normal((2, 2))
                       + 1j*RS.standard_normal((2, 2)))
    random_qutrit_op = (RS.standard_normal((3, 3))
                        + 1j*RS.standard_normal((3, 3)))

    dephase_proc_tensor = supops.process_to_proc_tensor(
            dephase_process, 2)
    depol_proc_tensor = supops.process_to_proc_tensor(
            depol_process, 2)
    qutrit_proc_tensor = supops.process_to_proc_tensor(
            random_qutrit_process, 3)
    proc_tensor_2_to_3 = supops.process_to_proc_tensor(
            process_2_to_3, 2)

    check_mat_eq(dephase_process(random_qubit_op),
                 supops.act_proc_tensor(random_qubit_op, dephase_proc_tensor))

    check_mat_eq(depol_process(random_qubit_op),
                 supops.act_proc_tensor(random_qubit_op, depol_proc_tensor))
    check_mat_eq(random_qutrit_process(random_qutrit_op),
                 supops.act_proc_tensor(random_qutrit_op, qutrit_proc_tensor))
    check_mat_eq(process_2_to_3(random_qubit_op),
                 supops.act_proc_tensor(random_qubit_op, proc_tensor_2_to_3))

    dephase_Ks = supops.proc_tensor_to_kraus_decomp(dephase_proc_tensor)
    depol_Ks = supops.proc_tensor_to_kraus_decomp(depol_proc_tensor)
    qutrit_Ks = supops.proc_tensor_to_kraus_decomp(qutrit_proc_tensor)
    Ks_2_to_3_orthog = supops.proc_tensor_to_kraus_decomp(proc_tensor_2_to_3)

    check_mat_eq(dephase_process(random_qubit_op),
                 act_kraus_ops(random_qubit_op, dephase_Ks))
    check_mat_eq(depol_process(random_qubit_op),
                 act_kraus_ops(random_qubit_op, depol_Ks))
    check_mat_eq(random_qutrit_process(random_qutrit_op),
                 act_kraus_ops(random_qutrit_op, qutrit_Ks))
    check_mat_eq(process_2_to_3(random_qubit_op),
                 act_kraus_ops(random_qubit_op, Ks_2_to_3_orthog))
