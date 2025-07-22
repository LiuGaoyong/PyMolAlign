import numpy as np
from scipy.spatial.transform import Rotation

from pymolalign.solver import NearCongruenceResult, molalign


def test_case():
    B = np.array(
        [
            [1.578385, 0.147690, 0.343809],
            [1.394750, 0.012968, 1.413545],
            [1.359929, -1.086203, -0.359782],
            [0.653845, 1.215099, -0.221322],
            [1.057827, 2.180283, 0.093924],
            [0.729693, 1.184864, -1.311438],
            [-0.817334, 1.152127, 0.208156],
            [-1.303525, 2.065738, -0.145828],
            [-0.883765, 1.159762, 1.299260],
            [1.984120, -1.734446, -0.021385],
            [2.616286, 0.458948, 0.206544],
            [-1.627725, -0.034052, -0.311301],
            [-2.684229, 0.151015, -0.118566],
            [-1.501868, -0.118146, -1.397506],
            [-1.324262, -1.260154, 0.333377],
            [-0.417651, -1.475314, 0.076637],
        ]
    )
    N = B.shape[0]
    R_true = Rotation.random()
    P_true = np.eye(N)[np.random.permutation(N)]  # Random permutation
    noise = 0.05 * np.random.randn(N, 3)  # Distortion noise
    A = P_true @ R_true.apply(B)
    print(A)
    print(B)
    print(A.mean(0))
    print(B.mean(0))

    res, msd = molalign(A, B)
    print("+" * 32)
    print(msd)
    print(res.T)
    print(res.P - P_true)
    print(res.R.as_matrix() - R_true.as_matrix())
    print("??", res.convert(B) - A)

    res0 = NearCongruenceResult(R_true, P_true, np.zeros(3))  # type: ignore
    print("-" * 32)
    print(res0.T)
    print(res0.P - P_true)
    print(res0.R.as_matrix() - R_true.as_matrix())
    print(res0.convert(B) - A)
