import numpy as np
from scipy.spatial.transform import Rotation

POS = [
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
POS = np.asarray(POS)


def test_case() -> None:
    assert isinstance(POS, np.ndarray)
    B, N = POS, POS.shape[0]

    R_true = Rotation.random()
    P_true = np.random.permutation(N)
    noise = 0.02 * np.random.normal(0, 0.01, (N, 3))
    A0 = R_true.apply(B)
    A1 = A0 + noise
    A2 = A0 + 1

    for A in (A0, A1, A2):
        R, rmsd = Rotation.align_vectors(A - A.mean(0), B - B.mean(0))  # type: ignore
        print(rmsd, (R.as_matrix() - R_true.as_matrix()).flatten())
        print(A - R.apply(B))
        print(A.mean(0), B.mean(0))
        print("*" * 32)
