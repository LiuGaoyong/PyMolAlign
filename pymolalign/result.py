import dataclasses as dc

import numpy as np
from scipy.spatial.transform import Rotation


@dc.dataclass(frozen=True)
class NearCongruenceResult:
    """The result of a Near-Congruence alignment.

    Eq:
        A = RBP + t
    Where:
        A: Reference structure
        B: The structure will be aligned
        R: Rotation matrix  (3x3 float matrix)
        P: Permutation array (N int array)
        t: Translation vector (3 float array)
    """

    rotation: Rotation
    permutation: np.ndarray  # N int
    translation: np.ndarray  # 3 float

    def __post_init__(self):
        """Post-initialization hook."""
        object.__setattr__(self, "R", self.rotation)
        if not isinstance(self.translation, tuple):
            trans_np = np.asarray(self.translation, dtype=float)
            if trans_np.shape != (3,):
                raise ValueError(f"Invalid translation: {self.translation}")
            object.__setattr__(self, "translation", trans_np)
        if not isinstance(self.permutation, tuple):
            perm_np = np.asarray(self.permutation, dtype=int)
            if perm_np.ndim == 2:
                perm_np = np.where(perm_np.astype(bool))[1]
            if perm_np.ndim != 1:
                raise ValueError(f"Invalid permutation: {self.permutation}")
            object.__setattr__(self, "permutation", perm_np.astype(int))

    @property
    def R(self) -> Rotation:
        """Rotation matrix."""
        return self.rotation

    @R.setter
    def R(self, value: Rotation):
        if not isinstance(value, Rotation):
            rot = np.asarray(value, dtype=float)
            if rot.shape == (3, 3):
                object.__setattr__(self, "rotation", Rotation.from_matrix(rot))
            elif rot.shape == (4,):
                object.__setattr__(self, "rotation", Rotation.from_quat(rot))
            else:
                raise ValueError(f"Invalid rotation matrix: {value}")

    @property
    def P(self) -> np.ndarray:
        """Permutation array."""
        result = np.zeros((len(self.permutation), len(self.permutation)))
        col = np.array(list(self.permutation), dtype=int)
        row = np.arange(len(col), dtype=int)
        result[row, col] = 1
        return result

    @P.setter
    def P(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=int)
        if value.ndim == 2:
            value = np.where(value.astype(bool))[1]
        if value.ndim != 1:
            raise ValueError(f"Invalid permutation: {value}")
        object.__setattr__(self, "permutation", value.astype(int))

    @property
    def T(self) -> np.ndarray:
        """Translation vector."""
        return np.array(list(self.translation), dtype=float)

    def convert(self, B: np.ndarray) -> np.ndarray:
        """Converts the aligned structure to the reference structure."""
        return self.R.apply(B)[self.P] + self.T
