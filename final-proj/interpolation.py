from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt


class BSpline:
    def __init__(
        self,
        t: Sequence[float], # knots
        c: Sequence[float], # control points
        d: int, # degree
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
        return(self.d == len(self.t) - len(self.c) - 1)
        return True  # change this.

    def bases(self, x: float, k: int, i: int) -> float:
        """
        Evaluate the B-spline basis function i, k at input position x.
        (Note that i, k start at 0.)
        """
        # TODO: complete this function

        if k == 0: # base case
            if self.t[i] <= x and self.t[i+1] > x:
                return 1
            else:
                return 0
            
        return (
            (x - self.t[i]) / (self.t[i + k - 1] - self.t[i]) * self.bases(x, i, k - 1) 
        ) + (
            (self.t[i + k] - x) / (self.t[i + k] - self.t[i + 1]) * self.bases(x, i+1, k - 1)
        )


    def interp(self, x: float) -> float:
        """Evaluate the B-spline at input position x."""
        # TODO: complete this function
        return self.bases(x, 0, self.d)
        return None  # change this


if __name__ == "__main__":
    t = [0, .2, .4, .6, .8, .1]  # set some knots. change this.
    c = [np.array([1, 0.2, 0.6]), np.array([0.2, 0.8, 0.2]), np.array([0.2, 0.8, 0.2])]  # set some control points (such as control colors). change this.
    d = 2  # set the degree.  change this.
    spline = BSpline(t, c, d)
    x = np.linspace(1.5, 4.5, 50)
    y = spline.interp(4.5)
    print(y)

    plt.plot(x, y)
    plt.title('B-Spline Curve')
    plt.xlabel('x')
    plt.ylabel('S(x)')
    plt.grid(True)
    plt.show()
    # now interpolate at some value
    #value = None  #  change this.
    #spline.interp(value)
