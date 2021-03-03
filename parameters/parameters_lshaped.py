from dolfin import (Expression, Constant, SubDomain, near, PETScKrylovSolver)
from PDE_Solver.tools import ParametersBase
import numpy as np


class Parameters(ParametersBase):
    domains = ['lconcave']
    meshes = ['01']

    def set_coefficients(self):
        #######################################################################
        # CONSTANTS
        self.T = 1
        self.alpha_range = [0, 1]  # market price of volatility risk
        self.control_set_size = 2  # number of available values of lambda

        self.howmaxit = 3  # maximum number of iterations of Howard's algorithm
        self.solver = PETScKrylovSolver('gmres', 'sor')
        self.solver.parameters['report'] = False
        self.solver.parameters['monitor_convergence'] = False
        self.solver.parameters['nonzero_initial_guess'] = True
        # interval at which output is saved to a chosen file i.e output will
        #  be saved every save_interval timesteps
        self.save_interval = 10  # save solution very 10th timestep
        if self.save_interval < 1:
            self.save_interval = 1

        #######################################################################
        # COEFFICIENTS OF IBVP

        # final time condition
        # REMARK: penalty for not leaving
        self.ft = Expression(
            'abs(x[0] - 1.0) < 1E-10 ? 0: 10.0', degree=1)

        # diffusion term
        # REMARK: remember that diffusion is non-negative
        def diffusion(alpha):
            return Expression(
                '(-0.05*(alpha)+0.05*(1-(1-alpha)*x[1]))',
                'alpha*(abs(x[0]-0.375)<0.05?0:0.1)+(1-alpha)*0.025',
                degree=1, alpha=alpha)
        self.__dict__['diffusion'] = diffusion

        # advection term in x direction

        def adv_x(alpha):
            return Expression(
                'abs(x[0]-0.375)<0.0625?0:-2.0*alpha', degree=1, alpha=alpha)
        self.__dict__['adv_x'] = adv_x

        # advection term in y direction
        def adv_y(alpha):
            return Expression(
                '-2.0*(1-alpha)', degree=1, alpha=alpha)
        self.__dict__['adv_y'] = adv_y

        # linear term
        def lin(alpha):
            return Constant('0.0')
        self.__dict__['lin'] = lin

        # advection terms in x direction on robin boundary
        self.robin_adv_x = {}

        def robin_adv_x_1(alpha):
            return Expression('0.0', degree=1, alpha=alpha)
        self.robin_adv_x[1] = robin_adv_x_1

        # lower left triangle
        def robin_adv_x_2(alpha):
            return Expression('1.0-2*alpha', degree=1, alpha=alpha)
            # return Expression('1.0', degree=1, alpha=alpha)  # Skorokhod
        self.robin_adv_x[2] = robin_adv_x_2

        # advection terms in y direction on robin time boundary
        self.robin_adv_y = {}

        def robin_adv_y_1(alpha):
            # return Expression('(1-alpha)*x[0]', degree=1, alpha=alpha)
            return Expression('0.0', degree=1, alpha=alpha)
        self.robin_adv_y[1] = robin_adv_y_1

        # lower left triangle
        def robin_adv_y_2(alpha):
            # return Expression('(1-alpha)*x[0]', degree=1, alpha=alpha)
            return Expression('-1.0', degree=1, alpha=alpha)
        self.robin_adv_y[2] = robin_adv_y_2

        # advection terms in y direction on robin time boundary
        self.robin_lin = {}

        def robin_lin_1(alpha):
            return Constant('0.0')
        self.robin_lin[1] = robin_lin_1
        self.robin_lin[2] = robin_lin_1

        # forcing term
        def RHSt(alpha):
            return Constant('0.0')
        self.__dict__['RHSt'] = RHSt

    def set_boundary_conditions(self, mesh):
        #######################################################################
        # BOUNDARY CONDITIONS
        # this assumes that domain is a rectangle

        right = np.amax([point[0] for point in mesh.coordinates()])

        # Dirichlet condition
        u_D = Constant('10.0')

        # Robin RHS
        def u_RR(alpha):
            return Constant('0.0')

        def u_RLLT(alpha):
            return Constant('0.0')

        tol = 1e-10

        class BoundaryDirichlet(SubDomain):
            def inside(self, x, on_boundary):
                return (on_boundary
                        and not near(x[0], right, tol)
                        and not ((0.25 - tol < x[0] < 0.5 - tol)
                                 and near(x[1], x[0]-0.25, tol)))

        class BoundaryRobinRight(SubDomain):
            def inside(self, x, on_boundary):
                return (on_boundary
                        and near(x[0], right, tol))

        class BoundaryRobinLowLeftTriangle(SubDomain):
            def inside(self, x, on_boundary):
                return ((0.25 - tol < x[0] < 0.5 - tol)
                        and near(x[1], x[0]-0.25, tol))
        self.omegas = {
            0: BoundaryDirichlet(),  # left, top and bottom
            1: BoundaryRobinRight(),  # right
            2: BoundaryRobinLowLeftTriangle()
        }

        self.regions = {
            "Dirichlet": [0],
            "Robin": [1, 2],
            "RobinTime": [],
        }

        self.RHS_bound = {
            0: u_D,
            1: u_RR,
            2: u_RLLT
        }

        self.time_dependent = {
            'rhs': False,
            'pde': False,
            'boundary': [],
        }
