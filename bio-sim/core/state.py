"""
発酵槽の状態を格納するデータモデルを提供する．
"""

from dataclasses import dataclass
import numpy as np

STATE_VARS: tuple[str, ...] = ("V", "X")
_NON_NEGATIVE_VARS: tuple[str, ...] = ("V", "X")


@dataclass
class ReactorState:
    """
    Reactor のある時刻における状態のスナップショット

    Attributes:
        t: 時刻 [h]
        V: 培養液量 [L]
        X: 菌体濃度 [g/L]
    """

    t: float
    V: float
    X: float

    def __post__init__(self) -> None:
        for name in _NON_NEGATIVE_VARS:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} は非負である必要があります(got {value})")

    def to_array(self) -> np.ndarray:
        """
        ODEソルバーに渡す配列形式に変換する
        """
        return np.array([getattr(self, name) for name in STATE_VARS], dtype=float)

    @classmethod
    def from_array(cls, t: float, y: np.ndarray) -> "ReactorState":
        """
        solve_ivp の積分結果（1時刻分の配列）からReactorStateを復元する
        """
        if len(y) != len(STATE_VARS):
            raise ValueError(
                f"y の長さは{len(STATE_VARS)}である必要があります (got {len(y)})"
            )
        kwargs = dict(zip(STATE_VARS, y))
        return cls(t=t, **kwargs)
