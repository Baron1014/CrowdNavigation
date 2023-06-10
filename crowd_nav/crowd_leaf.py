from rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation as Rot

from inference import inference


class CrowdAvoidanceLeaf(RMPLeaf):
    """
    Goal Attractor RMP leaf
    """

    def __init__(self, name, parent, y_g, w_u=10, w_l=1, sigma=1, alpha=1, eta=2, gain=1.0, tol=0.005):

        if y_g.ndim == 1:
            y_g = y_g.reshape(-1, 1)
        N = y_g.size
        psi = lambda y: (y - y_g)
        J = lambda y: np.eye(N)
        J_dot = lambda y, y_dot: np.zeros((N, N))

        def RMP_func(x, x_dot):

            G = np.eye(N)
            M = G
            f = inference(
                x[0],
                x[1],
                x_dot,
            )

            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)
