from pathlib import Path
import time
import json

from dolfin import (Mesh, FunctionSpace, TestFunction, TrialFunction,
                    DirichletBC, MeshFunction, Measure, assemble, dx,
                    Constant, dot, grad, interpolate, Function, action,
                    UnitIntervalMesh, XDMFFile, Point,
                    dof_to_vertex_map, vertex_to_dof_map, Vector)
import numpy as np

from PDE_Solver.tools import (getrows, calc_ad, apply_ad, set_zero_rows,
                              error_calc, matmult, toscipy, explicit_check,
                              implicit_check, get_min_timestep)

###############################################################################
# DEFINITION OF THE FINAL TIME BOUNDARY VALUE PROBLEM
###############################################################################


class FBVP:
    def __init__(self, mesh_name, parameters, alpha_range=None):
        # LOAD MESH AND PARAMETERS
        self.parameters = parameters
        self.mesh_name = mesh_name
        if not alpha_range:
            self.alpha_range = parameters.alpha_range
        else:
            self.alpha_range = alpha_range
        mesh_path = self.parameters.get_mesh_path()

        self.mesh = Mesh()
        with XDMFFile(mesh_path) as f:
            f.read(self.mesh)
        print(f'Mesh size= {self.mesh.hmax()}')
        # dimension of approximation space
        self.dim = len(self.mesh.coordinates())
        print(f'Dimension of solution space is {self.dim}')
        self.V = FunctionSpace(self.mesh, 'CG', 1)  # CG =  P1
        self.coords = self.mesh.coordinates()[dof_to_vertex_map(self.V)]
        self.T = self.parameters.T  # final time
        self.w = TrialFunction(self.V)
        self.u = TestFunction(self.V)
        #######################################################################
        # CONTROL SET CREATION
        self.control_set = np.linspace(self.alpha_range[0], self.alpha_range[1],
                                       self.parameters.control_set_size)
        self.control_set_size = len(self.control_set)
        print(f'Discretized control set has size {self.control_set_size}')
        #######################################################################
        # BOUNDARY CONDITIONS
        parameters.set_boundary_conditions(self.mesh)
        self.boundary_markers = MeshFunction('size_t', self.mesh, 1)
        self.boundary_markers.set_all(4)  # pylint: disable=no-member

        for i, omega in self.parameters.omegas.items():
            omega.mark(self.boundary_markers, i)

        self.ds = Measure('ds', domain=self.mesh,
                          subdomain_data=self.boundary_markers)

        self.dirichlet_bcs = [DirichletBC(self.V, parameters.RHS_bound[j],
                                          self.boundary_markers, j)
                              for j in self.parameters.regions["Dirichlet"]]

        # Get indices of dirichlet and robin dofs
        self.dirichlet_nodes_list = set()
        self.dirichlet_nodes_dict = {}
        for j in self.parameters.regions["Dirichlet"]:
            bc = DirichletBC(self.V, Constant(
                0), self.boundary_markers, j)
            self.dirichlet_nodes_list |= set(bc.get_boundary_values().keys())
            self.dirichlet_nodes_dict[j] = list(
                bc.get_boundary_values().keys())

        self.robin_nodes_list = set()
        self.robin_nodes_dict = {}
        for j in self.parameters.regions["Robin"]:
            bc = DirichletBC(self.V, Constant(
                0), self.boundary_markers, j)
            self.robin_nodes_list |= set(bc.get_boundary_values().keys())
            self.robin_nodes_dict[j] = list(bc.get_boundary_values().keys())

        bc = DirichletBC(self.V, Constant(0), 'on_boundary')
        self.boundary_nodes_list = bc.get_boundary_values().keys()

        self.robint_nodes_list = set()
        self.robint_nodes_dict = {}
        for j in self.parameters.regions["RobinTime"]:
            bc = DirichletBC(self.V, Constant(
                0), self.boundary_markers, j)
            self.robint_nodes_list |= set(bc.get_boundary_values().keys())
            self.robint_nodes_dict[j] = list(bc.get_boundary_values().keys())
        #######################################################################
        # ASSEMBLY
        time_start = time.process_time()
        self.assemble_diagonal_matrix()  # auxilliary generic diagonal matrix
        # used for vector*matrix multiplication of dolfin matrices
        self.assemble_lumpedmm()  # lumped mass matrix
        # which serves the role of identity operator
        self.assemble_laplacian()  # discrete laplacian

        self.ad_data_path = f'out/{self.parameters.experiment}'
        Path(self.ad_data_path).mkdir(parents=True, exist_ok=True)

        self.timesteps = self.parameters.get_number_of_timesteps()
        self.assemble_HJBe()  # assembly of explicit operators
        self.assemble_HJBi()  # assembly of implicit operators
        self.assemble_RHS()  # assembly of forcing term
        print('Final time assembly complete')
        print(f'Assembly took {time.process_time() - time_start} seconds')

        print('===========================================================')

    def assemble_diagonal_matrix(self):
        print("Assembling auxilliary diagonal matrix")
        """ Operator assembled to get the right sparameterssity pattern
        (non-zero terms can only exist on diagonal)
        """
        mesh3 = UnitIntervalMesh(self.dim)
        V3 = FunctionSpace(mesh3, "DG", 0)
        wid = TrialFunction(V3)
        uid = TestFunction(V3)
        self.diag_matrix = assemble(uid*wid*dx)
        self.diag_matrix.zero()

    def assemble_lumpedmm(self):
        """ Assembly lumped mass matrix - equal to 0 on robin boundary since
            there is no time derivative there.
        """
        print("Assembling lumped mass matrix")
        mass_form = self.w * self.u * dx
        mass_action_form = action(mass_form, Constant(1))
        self.MM_terms = assemble(mass_action_form)

        for n in self.robint_nodes_list:
            self.MM_terms[n] = 1.0
        for n in self.robin_nodes_list:
            self.MM_terms[n] = 0.0

        self.mass_matrix = assemble(mass_form)
        self.mass_matrix.zero()
        self.mass_matrix.set_diagonal(self.MM_terms)
        self.scipy_mass_matrix = toscipy(self.mass_matrix)

    def assemble_laplacian(self):
        print("Assembling laplacians")
        # laplacian discretisation
        self.laplacian = assemble(dot(grad(self.w), grad(self.u))*dx)

    def assemble_HJBe(self, t=None):
        """ Assembly explicit operator for every control in the control set,
        then apply artificial diffusion to all of them. Note that artificial
        diffusion is calculated differently on the boundary nodes.
        """
        self.explicit_matrices = np.empty(self.control_set_size, dtype=object)
        self.explicit_diffusion = np.empty(
            [self.control_set_size, self.dim])
        diagonal_vector = Vector(self.mesh.mpi_comm(), self.dim)
        global_min_timestep = float('inf')
        # Create explicit operator for each control in control set
        for k, alpha in enumerate(self.control_set):
            if k % 10 == 0:
                print(f'Assembling explicit matrix under control {k}'
                      f' out of {self.control_set_size}')
            # coefficients in the interior
            advection_x = self.parameters.adv_x(alpha)
            advection_y = self.parameters.adv_y(alpha)
            reaction = self.parameters.lin(alpha)

            if t is not None:
                advection_x.t = t
                advection_y.t = t
                reaction.t = t

            # Discretize PDE (set values on boundary rows to zero)
            b = np.array([advection_x, advection_y])
            E_interior_form = (np.dot(b, grad(self.w)) +
                               reaction * self.w)*self.u * dx
            explicit_matrix = assemble(E_interior_form)
            set_zero_rows(explicit_matrix, self.boundary_nodes_list)

            # Calculate diffusion necessary to make explicit operator monotone
            min_diffusion = np.zeros(explicit_matrix.size(0))
            for rows, row_num in getrows(explicit_matrix, self.laplacian,
                                         ignore=self.boundary_nodes_list):
                min_diffusion[row_num] = calc_ad(rows, row_num)
            self.explicit_diffusion[k] = min_diffusion

            discrete_diffusion = apply_ad(self.laplacian, min_diffusion)
            explicit_matrix += discrete_diffusion

            for j in self.parameters.regions['RobinTime']:
                self.set_directional_derivative(
                    explicit_matrix, region=j, nodes=self.robint_nodes_dict[j],
                    control=alpha, time=t)

            explicit_matrix.get_diagonal(diagonal_vector)
            current_min_timestep = get_min_timestep(
                diagonal_vector, self.MM_terms,
                self.dirichlet_nodes_list | self.robin_nodes_list)
            global_min_timestep = min(
                current_min_timestep, global_min_timestep)
            self.explicit_matrices[k] = toscipy(explicit_matrix)

        #######################################################################
        min_timesteps = int(self.T/global_min_timestep) + 1
        if not self.timesteps or self.timesteps < min_timesteps:
            self.timesteps = min_timesteps
        try:
            filename = (f'meshes/{self.parameters.domain}/'
                        f'Mvalues-{self.parameters.experiment}.json')
            with open(filename, 'r') as f:
                min_timesteps_dict = json.load(f)
        except FileNotFoundError:
            min_timesteps_dict = {}
        min_timesteps_dict[self.parameters.mesh_name] = self.timesteps
        with open(filename, 'w') as f:
            json.dump(min_timesteps_dict, f)

        self.timestep_size = self.T / self.timesteps  # time step size
        self.parameters.calculate_save_interval()
        self.explicit_matrices = self.timestep_size * self.explicit_matrices - \
            np.repeat(self.scipy_mass_matrix, self.control_set_size)

        print('Checking if the explicit operators satisfy'
              ' monotonicity conditions')
        for explicit_matrix in self.explicit_matrices:
            explicit_check(explicit_matrix, self.dirichlet_nodes_list)

    def assemble_RHS(self, t=None):
        """Assemble right hand side of the FBVP
        """
        print('Assembling RHS')
        self.forcing_terms = np.empty([self.control_set_size, self.dim])
        for i, alpha in enumerate(self.control_set):
            rhs = self.parameters.RHSt(alpha)
            if t is not None:
                rhs.t = t
            # Initialise forcing term
            F = np.array(assemble(rhs * self.u * dx)[:])

            for j in self.parameters.regions['RobinTime']:
                rhs = self.parameters.RHS_bound[j](alpha)
                if t is not None:
                    rhs.t = t
                bc = DirichletBC(self.V, rhs, self.boundary_markers, j)
                vals = bc.get_boundary_values()
                F[list(vals.keys())] = list(vals.values())

            for j in self.parameters.regions['Robin']:
                rhs = self.parameters.RHS_bound[j](alpha)
                if t is not None:
                    rhs.t = t
                bc = DirichletBC(self.V, rhs, self.boundary_markers, j)
                vals = bc.get_boundary_values()
                F[list(vals.keys())] = list(vals.values())

            self.forcing_terms[i] = self.timestep_size*F

    def assemble_HJBi(self, t=None):
        """ Assembly matrix discretizing second order terms after diffusion
        moved to the explicit operator is subtracted. Whenever amount of
        diffusion used to make some row of an explicit operator monotonic
        exceeds the amount of natural diffusion at that node we call it
        artificial diffusion. In such case, this row of the implicit operator
        is multiplied by zero
        """
        print('Assembling implicit operators')
        self.implicit_matrices = []
        remaining_diffusion = Function(self.V)
        max_art_dif = 0.0
        max_art_dif_loc = None
        diffusion_matrix = self.diag_matrix.copy()
        for explicit_diffusion, alpha in \
                zip(self.explicit_diffusion, self.control_set):

            diffusion = self.parameters.diffusion(alpha)
            if t is not None:
                diffusion.t = t

            diff = interpolate(diffusion, self.V).vector()
            if not np.all(diff >= 0):
                raise Exception("Choose non-negative diffusion")
            diff_vec = np.array([
                diff[i] if i not in self.boundary_nodes_list else 0.0
                for i in range(self.dim)])

            artificial_diffusion = explicit_diffusion - diff_vec
            if np.amax(artificial_diffusion) > max_art_dif:
                max_art_dif = np.amax(artificial_diffusion)
                max_art_dif_loc = self.coords[np.argmax(artificial_diffusion)]

            # discretise second order terms
            remaining_diffusion.vector(
            )[:] = np.maximum(-artificial_diffusion, [0]*self.dim)
            diffusion_matrix.set_diagonal(remaining_diffusion.vector())
            implicit_matrix = matmult(diffusion_matrix, self.laplacian)

            for j in self.parameters.regions['Robin']:
                self.set_directional_derivative(
                    implicit_matrix, region=j, nodes=self.robin_nodes_dict[j],
                    control=alpha, time=t)

            self.implicit_matrices.append(
                self.timestep_size * implicit_matrix + self.mass_matrix)

        self.scipy_implicit_matrices = [
            toscipy(mat) for mat in self.implicit_matrices]
        print('Checking if the implicit operators satisfy'
              ' monotonicity conditions')
        for implicit_matrix in self.scipy_implicit_matrices:
            implicit_check(implicit_matrix)

        with open(self.ad_data_path+'/ad.txt', 'a') as f:
            time_str = f' at time {t}\n' if t is not None else '\n'
            f.write(f'For mesh {self.mesh_name} max value of artificial'
                    ' diffusion coefficient was'
                    f' {max_art_dif} at {max_art_dif_loc}'
                    + time_str)

    def set_directional_derivative(self, operator, region, nodes, control,
                                   time=None):
        if region in self.parameters.regions['Robin']:
            adv_x = self.parameters.robin_adv_x[region](control)
            adv_y = self.parameters.robin_adv_y[region](control)
            lin = self.parameters.robin_lin[region](control)
        elif region in self.parameters.regions['RobinTime']:
            adv_x = self.parameters.robint_adv_x[region](control)
            adv_y = self.parameters.robint_adv_y[region](control)
            lin = self.parameters.robint_lin[region](control)

        if time is not None:
            adv_x.t = time
            adv_y.t = time
            lin.t = time
        b = (interpolate(adv_x, self.V),
             interpolate(adv_y, self.V))
        c = interpolate(lin, self.V)
        for n in nodes:
            # node coordinates
            x = self.coords[n]
            # evaluate advection at robin node
            b_x = np.array([b[0].vector()[n], b[1].vector()[n]])
            # denominator used to calculate directional derivative
            if np.linalg.norm(b_x) > 1:
                lamb = 0.1*self.mesh.hmin()/np.linalg.norm(b_x)
            else:
                lamb = 0.1*self.mesh.hmin()
            # position of first node of the stencil
            x_prev = x - lamb * b_x
            # Find cell containing first node of stencil and get its
            # dof/vertex coordinates
            try:
                cell_ind = self.mesh.bounding_box_tree(
                ).compute_entity_collisions(Point(x_prev))[0]
            except IndexError:
                i = 16
                while i > 2:
                    # sometimes Fenics does not detect nodes if boundary is
                    # parallel to the boundary advection due to rounding errors
                    # so try different precisions just to be sure
                    try:
                        cell_ind = self.mesh.bounding_box_tree(
                        ).compute_entity_collisions(
                            Point(np.round(x_prev, i)))[0]
                        break
                    except IndexError:
                        i -= 1
                else:
                    raise Exception(
                        "Boundary advection outside tangential cone")
            cell_vertices = self.mesh.cells()[cell_ind]
            cell_dofs = vertex_to_dof_map(self.V)[cell_vertices]
            cell_coords = self.mesh.coordinates()[cell_vertices]
            # calculate weigth of each vertex in the cell (using
            #  barycentric coordinates)
            A = np.vstack((cell_coords.T, np.ones(3)))
            rhs = np.append(x_prev, np.ones(1))
            weights = np.linalg.solve(A, rhs)
            weights = [w if w > 1e-14 else 0 for w in weights]
            dof_to_weight = dict(zip(cell_dofs, weights))

            # calculate directional derivative at each node using
            # weights to interpolate value of numerical solution at
            # x_prev
            row = operator.getrow(n)
            indices = row[0]
            data = row[1]
            for dof in cell_dofs:
                pos = np.where(indices == dof)[0][0]
                if dof != n:
                    data[pos] = - dof_to_weight[dof] / lamb
                else:
                    c_n = c.vector()[dof]
                    # make sure reaction term is positive adding artificial
                    # constant if necessary
                    if region in self.parameters.regions['Robin']:
                        c_n = max(c_n, min(lamb, 1E-8))
                    data[pos] = (1-dof_to_weight[dof]) / lamb + c_n
            operator.set([data], [n], indices)
            operator.apply('insert')
###############################################################################
# SOLVER
###############################################################################


class Solver:
    def __init__(self, fbvp, howard_inf=True, return_result=False,
                 output_mode=True, get_error=False, save_dir=None,
                 linear_mode=False, verbose=True, record_control=True):
        self.fbvp = fbvp
        self.solver = fbvp.parameters.solver
        self.howard_inf = howard_inf
        self.output_mode = output_mode
        self.get_error = get_error
        self.return_result = return_result
        self.linear_mode = linear_mode
        self.record_control = record_control
        if self.linear_mode:
            self.linear_control = 0
        self.verbose = verbose
        # Initialize solution vector with final time data
        self.v = interpolate(fbvp.parameters.ft, fbvp.V)
        self.control = interpolate(Constant(0), fbvp.V)
        if not save_dir:
            save_dir = f'out/{fbvp.parameters.experiment}/'\
                       f'{fbvp.parameters.domain}/{fbvp.mesh_name}/v.xdmf'
        if self.output_mode:
            self.file = XDMFFile(save_dir)
            self.file.parameters['rewrite_function_mesh'] = False
            self.file.write_checkpoint(self.v, 'value_func', fbvp.parameters.T)

        print('Ready to begin calculation')

    def time_iter(self):
        alpha = [0]*self.fbvp.dim  # initialize control
        for k in range(self.fbvp.timesteps - 1, -1, -1):
            time_start = time.perf_counter()
            t = k * self.fbvp.timestep_size  # Current time

            # update dirichlet boundary conditions if necessary
            # TODO: Similar update for Robin boundaries
            if any(elem in self.fbvp.parameters.time_dependent['boundary']
                   for elem in self.fbvp.parameters.regions['Dirichlet']):
                self.update_dirichlet_boundary(t)

            if self.fbvp.parameters.time_dependent['rhs']:
                self.fbvp.assemble_RHS(t)

            if self.fbvp.parameters.time_dependent['pde']:
                self.fbvp.assemble_HJBe(t+self.fbvp.timestep_size)
                self.fbvp.assemble_HJBi(t)

            if self.linear_mode:
                A = self.fbvp.Ilist[self.linear_control]
                b = self.v.vector().copy()
                b[:] = (self.fbvp.Flist[self.linear_control] -
                        self.fbvp.Elist[self.linear_control]
                        .dot(np.array(self.v.vector()[:])))
                for bc in self.fbvp.dirichlet_bcs:
                    bc.apply(A, b)
                howit = self.v.vector().copy()
                self.solver.solve(A, howit, b)
            else:
                ell = 0  # iteration counter
                while ell < self.fbvp.parameters.howmaxit:
                    ell += 1
                    howit, alpha = self.Howard(self.v.vector(), alpha)

            self.v.vector()[:] = howit

            if self.output_mode and k % self.fbvp.parameters.save_interval == 0:
                self.file.write_checkpoint(
                    self.v, 'value_func', t, XDMFFile.Encoding.HDF5, True)

                if self.record_control:
                    self.control.vector()[:] = alpha
                    self.file.write_checkpoint(
                        self.control, 'control', t,
                        XDMFFile.Encoding.HDF5, True)

            time_elapsed = (time.perf_counter()-time_start)
            if self.verbose:
                print(
                    f'Time step {self.fbvp.timesteps - k} out of'
                    f' {self.fbvp.timesteps} took {time_elapsed}')

        if self.return_result:
            return self.v

        if self.get_error:
            error_file_path = (f'out/{self.fbvp.parameters.experiment}/'
                               f'{self.fbvp.parameters.domain}/errors.json')
            error_calc(self.v, self.fbvp.parameters.v_e, self.fbvp.mesh,
                       self.fbvp.mesh_name,
                       error_file_path)

    def Howard(self, v, alpha):
        # v coresponds to v^{k+1}, save it to numpy array
        spv = np.array(v[:])
        Ev = [E.dot(spv) for E in self.fbvp.explicit_matrices]
        # construct RHS under input control alfa
        rhs = v.copy()
        rhs[:] = [self.fbvp.forcing_terms[control][i]-Ev[control][i]
                  for i, control in enumerate(alpha)]
        # initialise vector with correct dimension to store solution
        how = v.copy()

        # initalise matrix with a suitable sparameterssity pattern
        lhs = self.fbvp.implicit_matrices[0].copy()
        # construct implicit matrix from control alpha
        for i, control in enumerate(alpha):
            lhs.set([self.fbvp.implicit_matrices[control].getrow(i)[1]], [i],
                    self.fbvp.implicit_matrices[control].getrow(i)[0])
        lhs.apply('insert')

        # solve linear problem to get next iterate of u
        for bc in self.fbvp.dirichlet_bcs:
            bc.apply(lhs, rhs)
        self.solver.solve(lhs, how, rhs)

        # Create list of vectors (I*v^k+E*v^{k+1} -F) under different controls
        spw = np.array(how[:])
        multlist = (Imp.dot(spw) + Exp - F
                    for Imp, Exp, F in
                    zip(self.fbvp.scipy_implicit_matrices,
                        Ev, self.fbvp.forcing_terms))

        # Loop over vectors of values of (I*v^k+E*v^{k+1} -F) at each node and
        # record control which optimizes each of them
        if self.howard_inf:
            next_ctr = [np.argmin(vector) for vector in zip(*multlist)]
        else:
            next_ctr = [np.argmax(vector) for vector in zip(*multlist)]
        return (how, next_ctr)

    def update_dirichlet_boundary(self, t):
        new_dirichlet_bcs = []
        for i, region in enumerate(self.fbvp.parameters.regions['Dirichlet']):
            if region in self.fbvp.parameters.time_dependent['boundary']:
                new_dirichlet_bcs.append(
                    self.fbvp.parameters.update_dispenser[region](self.fbvp, t))
            else:
                new_dirichlet_bcs.append(self.fbvp.dirichlet_bcs[i])
        self.fbvp.dirichlet_bcs = new_dirichlet_bcs
