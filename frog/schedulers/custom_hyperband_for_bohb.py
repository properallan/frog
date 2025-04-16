from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
import numpy as np
from typing import Optional


class CustomHyperBandForBOHB(HyperBandForBOHB):
    def __init__(
        self,
        time_attr: str = "training_iteration",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        max_t: int = 81,
        reduction_factor: float = 3,
        stop_last_trials: bool = True,
        min_t: Optional[int] = None,
    ):
        super().__init__(
            time_attr=time_attr,
            metric=metric,
            mode=mode,
            max_t=max_t,
            reduction_factor=reduction_factor,
            stop_last_trials=stop_last_trials,
        )

        self.min_t = min_t

        if min_t is not None:
            # Redefinindo _get_r0 para usar o valor customizado como mínimo
            self._get_r0 = lambda s: int(self.min_t)