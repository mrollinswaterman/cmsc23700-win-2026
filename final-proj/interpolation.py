from typing import Sequence
import numpy as np


class BSpline:
    def __init__(
        self,
        t: Sequence[float],
        c: Sequence[float],
        d: int,
    ):
        """
        t = knots
        c = bspline coefficients / control points
        d = bspline degree
        """
        self.t = t
        self.c = c
        self.d = d
        assert self.is_valid()

    def is_valid(self) -> bool:
        """Check if the B-spline configuration is valid."""
        # TODO: complete this function.
        return True  # change this.

    def bases(self, x: float, k: int, i: int) -> float:
        """
        Evaluate the B-spline basis function i, k at input position x.
        (Note that i, k start at 0.)
        """
        # TODO: complete this function
        return None  # change this.

    def interp(self, x: float) -> float:
        """Evaluate the B-spline at input position x."""
        # TODO: complete this function
        return None  # change this


if __name__ == "__main__":
    t = []  # set some knots. change this.
    c = []  # set some control points (such as control colors). change this.
    d = None  # set the degree.  change this.
    spline = BSpline(t, c, d)
    # now interpolate at some value
    value = None  #  change this.
    spline.interp(value)
