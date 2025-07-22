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
        t: Translation vector    (3 float array)
    """

    rotation: Rotation
    permutation: tuple[int, ...]
    translation: tuple[float, float, float]

    def __post_init__(self):
        """Post-initialization hook."""
        if not isinstance(self.rotation, Rotation):
            rot = np.asarray(self.rotation, dtype=float)
            if rot.shape == (3, 3):
                object.__setattr__(self, "rotation", Rotation.from_matrix(rot))
            elif rot.shape == (4,):
                object.__setattr__(self, "rotation", Rotation.from_quat(rot))
            else:
                raise ValueError(f"Invalid rotation matrix: {self.rotation}")
        if not isinstance(self.translation, tuple):
            trans_np = np.asarray(self.translation)
            trans = tuple(float(x) for x in trans_np)
            object.__setattr__(self, "translation", trans)
        if not isinstance(self.permutation, tuple):
            perm_np = np.asarray(self.permutation, dtype=int)
            if perm_np.ndim == 2:
                perm_np = np.where(perm_np.astype(bool))[1]
            if perm_np.ndim == 1:
                perm = tuple(int(x) for x in perm_np)
            else:
                raise ValueError(f"Invalid permutation: {self.permutation}")
            object.__setattr__(self, "permutation", perm)

    @property
    def R(self) -> Rotation:
        """Rotation matrix."""
        return self.rotation

    @property
    def P(self) -> np.ndarray:
        """Permutation array."""
        result = np.zeros((len(self.permutation), len(self.permutation)))
        col = np.array(list(self.permutation), dtype=int)
        row = np.arange(len(col), dtype=int)
        result[row, col] = 1
        return result

    @property
    def T(self) -> np.ndarray:
        """Translation vector."""
        return np.array(list(self.translation), dtype=float)

    def convert(self, B: np.ndarray) -> np.ndarray:
        """Converts the aligned structure to the reference structure."""
        return self.P @ self.R.apply(B) + self.T
