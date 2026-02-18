from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt


class BSpline:
    def __init__(
        self,
        t: Sequence[float],  # knots
        c: Sequence[float],  # control points
        d: int,  # degree
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
        return self.d == len(self.t) - len(self.c) - 1


    def bases(self, x: float, k: int, i: int) -> float:
        """
        Evaluate the B-spline basis function i, k at input position x.
        (Note that i, k start at 0.)
        """
        if k == 0:  # base case
            # print("base case!")
            if self.t[i] <= x and self.t[i + 1] > x:
                return 1
            else:
                return 0
        else:
            # print("recurring...")
            return (
                (x - self.t[i]) / (self.t[i + k] - self.t[i]) * self.bases(x, k - 1, i)
            ) + (
                (self.t[i + k + 1] - x)
                / (self.t[i + k + 1] - self.t[i + 1])
                * self.bases(x, k - 1, i + 1)
            )

    def interp(self, x: float) -> float:
        """Evaluate the B-spline at input position x."""
        return self.bases(x=x, k=self.d, i=0)


if __name__ == "__main__":
    t = [0, 1, 2, 3, 4]  # set some knots. change this.
    c = [0, 3]  # set some control points (such as control colors). change this.
    d = 2  # set the degree.  change this.
    spline = BSpline(t, c, d)
    # now interpolate at some value
    x = np.linspace(0, 3)
    y = [spline.interp(v) for v in x]

    plt.plot(x, y)
    plt.title("B-Spline Curve")
    plt.xlabel("x")
    plt.ylabel("S(x)")
    plt.grid(True)
    plt.show()
