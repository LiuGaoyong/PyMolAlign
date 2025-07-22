from typing import Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from pymolalign.result import NearCongruenceResult


class NearCongruenceSolver:
    """The solver of a Near-Congruence alignment.

    Eq:
        A = RBP + t
    Where:
        A: Reference structure
        B: The structure will be aligned
        R: Rotation matrix  (3x3 float matrix)
        P: Permutation array (N int array)
        t: Translation vector    (3 float array)
    """

    def __init__(
        self,
        biasscale=1000.0,
        epsilon=0.15,
        maxiter=100,
        tol=1e-6,
    ) -> None:
        """Initializes the Near-Congruence Solver for rigid structures.

        Parameters:
            biasscale (float): Penalty value for infeasible assignments
            epsilon (float): Tolerance threshold for feasible assignments
                Note: Tolerance threshold ϵ: it is dependent on the
                    maximum atomic displacement u, ϵ≈2*3**0.5*u.
                Example: If u = 0.1 Å, then ε ≈ 0.35 Å
                Importance:
                    For u ≤ 0.1 Å, biased methods are 100,000x faster !
                    For u = 0.2 Å, performance drops to unbiased levels.
            maxiter (int): Maximum iterations of main loop
            tol (float): Convergence tolerance for MSD
        """
        self.biasscale: float = float(biasscale)
        self.epsilon: float = float(epsilon)
        self.maxiter: int = int(maxiter)
        self.tol: float = float(tol)

    @staticmethod
    def _check_coordinates(points: np.ndarray) -> np.ndarray:
        """Checks the coordinates of the points."""
        points = np.asarray(points, dtype=float)
        if points.shape[0] < 3:
            raise ValueError("The number of points must be greater than 2.")
        if points.ndim != 2:
            raise ValueError("The number of dimensions must be 2.")
        if points.shape[1] != 3:
            raise ValueError("The points must be Nx3 shape.")
        return points

    @classmethod
    def _compute_euclidean_matrix(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        R: Union[np.ndarray, Rotation],
    ) -> np.ndarray:
        """Computes the squared Euclidean distance matrix."""
        A = cls._check_coordinates(A)
        B = cls._check_coordinates(B)
        if not isinstance(R, Rotation):
            rot = np.asarray(R, dtype=float)
            if rot.shape == (3, 3):
                R = Rotation.from_matrix(rot)
            elif rot.shape == (4,):
                R = Rotation.from_quat(rot)
            else:
                raise ValueError(f"Invalid rotation matrix: {R}")
        assert isinstance(R, Rotation)
        return cdist(A, R.apply(B), metric="sqeuclidean")

    @staticmethod
    def _solve_assignment(cost: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve Linear Assignment Problem (LAP) by modified-JV algorithm."""
        cost = np.asarray(cost, dtype=float)  # convert to numpy array
        assert cost.ndim == 2 and cost.shape[0] == cost.shape[1]
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build permutation matrix
        P = np.zeros_like(cost)
        P[row_ind, col_ind] = 1

        # Calculate MSD
        msd = np.sum(cost[row_ind, col_ind]) / cost.shape[0]

        return P, msd

    def _compute_bias_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Generates the bias matrix using the diagonal tolerance criterion.

        Parameters:
            A (np.ndarray): Nx3 coordinate matrix
            B (np.ndarray): Nx3 coordinate matrix

        Returns:
            np.ndarray: NxN bias matrix with penalties
        """
        A = self._check_coordinates(A)
        B = self._check_coordinates(B)

        # Compute sorted diagonal elements of AᵀA and BᵀB
        diag_A = np.sort(np.sum(A**2, axis=1))
        diag_B = np.sort(np.sum(B**2, axis=1))

        # Apply bias criterion: |diag_A[i] - diag_B[j]| > ε
        diff = np.abs(diag_A[:, None] - diag_B[None, :])
        F = np.where(diff <= self.epsilon, 0.0, self.biasscale)

        return F

    def _assignment_step(
        self,
        A: np.ndarray,
        B: np.ndarray,
        R: Union[np.ndarray, Rotation],
        bias_matrix: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """Solves the assignment problem with biased Euclidean costs.

        Returns:
            tuple: (Permutation matrix, MSD value)
        """
        A = self._check_coordinates(A)
        B = self._check_coordinates(B)

        E = self._compute_euclidean_matrix(A, B, R)
        if bias_matrix is not None:
            E += np.asarray(bias_matrix, dtype=float)

        return self._solve_assignment(E)

    def _alignment_step(
        self,
        A: np.ndarray,
        B: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[Rotation, float]:
        """Solves the optimal rotation via Kabsch algorithm.

        Parameters:
            A (np.ndarray): Nx3 coordinate matrix
            B (np.ndarray): Nx3 coordinate matrix
            weights (np.ndarray, optional): the weights of the points.

        Returns:
            tuple: (Rotation object, RMSD value)
        """
        A = self._check_coordinates(A)
        B = self._check_coordinates(B)

        rot, rmsd = Rotation.align_vectors(  # type: ignore
            A,
            B,
            weights=weights,
            return_sensitivity=False,
        )
        return rot, rmsd

    def __call__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        weights: Optional[np.ndarray] = None,
        use_bias: bool = True,
    ) -> tuple[NearCongruenceResult, float]:
        """Main solver for near-congruence problem.

        Parameters:
            A (np.ndarray): Reference Nx3 coordinate matrix
            B (np.ndarray): Target Nx3 coordinate matrix
            weights (np.ndarray, optional): the weights of the points.
            use_bias (bool): Whether to use bias matrix or not

        Returns (tuple[NearCongruenceResult, float]):
            Solution containing rotation, permutation, and transition.
            The second float is the final MSD value.
        """
        CENTER_A = A.mean(0)
        CENTER_B = B.mean(0)
        t = CENTER_A - CENTER_B  # translation vector
        A, B = A - CENTER_A, B - CENTER_B

        msd = best_msd = float("inf")
        P = best_P = np.eye(A.shape[0])  # identity permutation
        F = self._compute_bias_matrix(A, B) if use_bias else None
        R, rmsd = best_R, best_rmsd = self._alignment_step(A, B, weights)

        # Iterative assignment & alignment loop
        for _ in range(self.maxiter):
            # Assignment step with biased costs
            P, msd = self._assignment_step(A, B, R, F)  # type: ignore
            B_permuted = P @ B

            # Alignment step via SVD
            R, rmsd = self._alignment_step(A, B_permuted, weights)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_R = R
                best_P = P
            print("???", _, rmsd, best_rmsd)

        result = NearCongruenceResult(best_R, best_P, t)  # type: ignore
        return result, best_msd


def molalign(
    ref_positions: np.ndarray,
    target_positions: np.ndarray,
    atomicmasses: Optional[np.ndarray] = None,
    biasscale: float = 1000.0,
    epsilon: float = 0.15,
    usebias: bool = True,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> tuple[NearCongruenceResult, float]:
    """Main solver for near-congruence problem.

    Parameters:
        ref_positions (np.ndarray): Reference Nx3 coordinate matrix
        target_positions (np.ndarray): Target Nx3 coordinate matrix
        atomicmasses (np.ndarray, optional): the atomic masses of the points.
        use_bias (bool): Whether to use bias matrix or not
        biasscale (float): Penalty value for infeasible assignments
        epsilon (float): Tolerance threshold for feasible assignments
            Note: Tolerance threshold ϵ: it is dependent on the
                maximum atomic displacement u, ϵ≈2*3**0.5*u.
            Example: If u = 0.1 Å, then ε ≈ 0.35 Å
            Importance:
                For u ≤ 0.1 Å, biased methods are 100,000x faster !
                For u = 0.2 Å, performance drops to unbiased levels.
        maxiter (int): Maximum iterations of main loop
        tol (float): Convergence tolerance for MSD

    Returns (tuple[NearCongruenceResult, float]):
        Solution containing rotation, permutation, and transition.
        The second float is the final MSD value.
    """
    return NearCongruenceSolver(
        biasscale=biasscale,
        epsilon=epsilon,
        maxiter=maxiter,
        tol=tol,
    )(
        A=ref_positions,
        B=target_positions,
        weights=atomicmasses,
        use_bias=usebias,
    )
