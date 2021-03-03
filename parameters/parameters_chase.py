import numpy as np

from dolfin import (PETScKrylovSolver, Expression, Constant, SubDomain)

from PDE_Solver.tools import ParametersBase


class Parameters(ParametersBase):
    domains = ['donut']
    meshes = ['1']

    def set_coefficients(self):
        self.T = 1

        self.alpha_range = [0, 2*np.pi]
        self.alpha_size = 10

        self.beta_range = [0, 2*np.pi]
        self.beta_size = 10

        self.howmaxit = 3  # maximum number of iterations of Howard's algorithm
        self.solver = PETScKrylovSolver('gmres', 'sor')
        self.solver.parameters['report'] = False
        self.solver.parameters['absolute_tolerance'] = 1E-12
        self.solver.parameters['relative_tolerance'] = 1E-8
        self.solver.parameters['monitor_convergence'] = False
        self.solver.parameters['nonzero_initial_guess'] = True

        def diffusion(alpha, beta):
            return Expression('x[1]<0.0?0.1:x[1]', degree=1)
        self.__dict__['diffusion'] = diffusion

        velocity_evader_hor = 0.5
        velocity_pursuer_hor = 4

        velocity_evader_ver = 1
        velocity_pursuer_ver = 1

        def adv_x(alpha, beta):
            exp = '(vP*sin(beta)-vE*sin(alpha))'
            return Expression(exp, degree=1,
                              alpha=alpha, beta=beta,
                              vE=velocity_evader_hor, vP=velocity_pursuer_hor)
        self.__dict__['adv_x'] = adv_x

        def adv_y(alpha, beta):
            exp = '(vP*cos(beta)-vE*cos(alpha))'
            return Expression(exp, degree=1,
                              alpha=alpha, beta=beta,
                              vE=velocity_evader_ver, vP=velocity_pursuer_ver)
        self.__dict__['adv_y'] = adv_y

        def lin(alpha, beta):
            return Constant('0.0')
        self.__dict__['lin'] = lin

        self.small_radius = 1
        self.large_radius = 4

        self.ft = Expression(
            'abs(x[0]*x[0] + x[1]*x[1] - r) < 1E-10 ? 0: 1.0',
            degree=1, r=self.small_radius)

        def RHSt(alpha, beta):
            return Constant('0.0')
        self.__dict__['RHSt'] = RHSt

    def set_boundary_conditions(self, mesh):

        small_radius = self.small_radius
        large_radius = self.large_radius

        hmin = mesh.hmin()

        class BoundaryDirichletInner(SubDomain):
            def inside(self, x, on_boundary):
                return (on_boundary and np.sqrt(
                    x[0]**2+x[1]**2) < (small_radius + hmin))

        class BoundaryDirichletOuter(SubDomain):
            def inside(self, x, on_boundary):
                return (on_boundary and np.sqrt(
                    x[0]**2+x[1]**2) > (large_radius - hmin))

        self.omegas = {
            0: BoundaryDirichletInner(),
            1: BoundaryDirichletOuter(),
        }

        g_inner = Constant('0.0')
        g_outer = Constant('1.0')

        self.RHS_bound = {
            0: g_inner,
            1: g_outer,
        }

        self.time_dependent = {
            'pde': False,
            'rhs': False,
            'boundary': [],
        }
