from typing import Sequence
import numpy as np


"""
BSpline methodology from here:
    - https://github.com/Liam-Xander/Simple-BSpline-Python/blob/main/Bsplne.py
"""

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
        if k == 0:
            if (x >= self.t[i]) and (x < self.t[i + 1]):
                return 1.0
            else:
                return 0.0
        else:

            length1 = self.t[i + k] - self.t[i]
            length2 = self.t[i + k + 1] - self.t[i + 1]

            if length1 == 0.0:
                length1 = 1.0
            if length2 == 0.0:
                length2 = 1.0

        term1 = (x - self.t[i]) / length1 * self.bases(x=x, k=k - 1, i=i)
        term2 = (self.t[i + k + 1] - x) / length2 * self.bases(x=x, k=k - 1, i=i + 1)

        return term1 + term2

    def interp(self, x: float) -> float:
        """Evaluate the B-spline at input position x."""
        #print("running interp...")
        sum = 0.0
        for i in range(len(self.c)):
            sum += self.c[i] * self.bases(x=x, k=self.d, i=i)
        return sum



if __name__ == "__main__":
    t = [0, 1, 2, 3, 4, 5]  # set some knots. change this.
    c = [0.6, 1, 0.2]  # set some control points (such as control colors). change this.
    d = 2  # set the degree.  change this.
    spline = BSpline(t, c, d)
    # now interpolate at some value
    x = 0.6122448979591836
    print(spline.interp(x))

    # plt.plot(x, y)
    # plt.title("B-Spline Curve")
    # plt.xlabel("x")
    # plt.ylabel("S(x)")
    # plt.grid(True)
    # plt.show()
