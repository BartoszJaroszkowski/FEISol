from pathlib import Path
import time
import json

from dolfin import (Mesh, XDMFFile, FunctionSpace, dof_to_vertex_map,
                    TrialFunction, TestFunction, MeshFunction, Measure,
                    DirichletBC, Constant, UnitIntervalMesh, assemble, dx,
                    action, dot, grad, Vector, Function, interpolate)
import numpy as np

from PDE_Solver.tools import (calc_ad, getrows, apply_ad, matmult,
                              get_min_timestep, toscipy, explicit_check,
                              implicit_check, error_calc)


class FBVP:
    def __init__(self, mesh_name, parameters,
                 alpha_range=None, beta_range=None):
        # DEFINE MESH AND PARAMETERS
        self.parameters = parameters
        self.mesh_name = mesh_name
        if not alpha_range:
            self.alpha_range = parameters.alpha_range
        else:
            self.alpha_range = alpha_range
        if not beta_range:
            self.beta_range = parameters.beta_range
        else:
            self.beta_range = alpha_range
        meshname = \
            f'meshes/{parameters.domain}/{parameters.domain}_{self.mesh_name}.xdmf'

        self.mesh = Mesh()
        with XDMFFile(meshname) as f:
            f.read(self.mesh)
        print(f'Mesh size= {self.mesh.hmax()}')
        # dimension of approximation space
        self.dim = len(self.mesh.coordinates())
        print(f'Dimension of solution space is {self.dim}')
        self.V = FunctionSpace(self.mesh, 'CG', 1)  # CG =  P1
        self.coords = self.mesh.coordinates()[dof_to_vertex_map(self.V)]
        # self.t_V = FunctionSpace(self.t_mesh, 'CG', 1)
        self.T = self.parameters.T  # final time
        self.w = TrialFunction(self.V)
        self.u = TestFunction(self.V)
        #######################################################################
        # CONTROL SET CREATION
        self.ctrset_alpha = np.linspace(self.alpha_range[0],
                                        self.alpha_range[1],
                                        parameters.alpha_size)
        self.ctrset_beta = np.linspace(self.beta_range[0],
                                       self.beta_range[1],
                                       parameters.beta_size)
        self.csize_alpha = len(self.ctrset_alpha)
        self.csize_beta = len(self.ctrset_beta)
        print(f'Discretized control set has size'
              f' {self.csize_alpha * self.csize_beta}')
        #######################################################################
        # BOUNDARY CONDITIONS
        parameters.set_boundary_conditions(self.mesh)
        self.boundary_markers = MeshFunction('size_t', self.mesh, 1)
        self.boundary_markers.set_all(99)  # pylint: disable=no-member

        for i, omega in self.parameters.omegas.items():
            omega.mark(self.boundary_markers, i)

        self.ds = Measure('ds', domain=self.mesh,
                          subdomain_data=self.boundary_markers)

        self.dirichlet_bcs = [DirichletBC(self.V, parameters.RHS_bound[i],
                                          self.boundary_markers, i)
                              for i in parameters.omegas]

        # Get indices of dirichlet and dofs
        self.boundary_nodes_list = \
            DirichletBC(self.V, Constant('0.0'),
                        'on_boundary').get_boundary_values().keys()
        #######################################################################
        # ASSEMBLY
        time_start = time.process_time()
        self.assemble_diagonal_matrix()  # auxilliary generic diagonal matrix
        # used for vector*matrix multiplication of dolfin matrices
        self.assemble_lumpedmm()  # assembly lumped mass matrix
        # which serves the role of identity operator
        self.assemble_laplacian()  # assembly of discrete laplacian

        self.ad_data_path = f'out/{parameters.experiment}'
        Path(self.ad_data_path).mkdir(parents=True, exist_ok=True)

        # FIXME: value of timestep required to impose monotonity may be
        # different at different times, at this moment code assumes that
        # it won't be smaller than in case t=T
        self.timesteps = parameters.get_number_of_timesteps()
        self.assemble_HJBe()  # assembly of explicit operators
        self.assemble_HJBi()  # assembly of implicit operators
        self.assemble_RHS()  # assembly of forcing term

        print('Final time assembly complete')
        print(f'Assembly took {time.process_time() - time_start} seconds')
        print('===========================================================')

    def assemble_diagonal_matrix(self):
        print("Assembling auxilliary diagonal matrix")
        """ Operator assembled to get the right sparsity pattern
            (non-zero terms can only exist on diagonal)
            """
        mesh3 = UnitIntervalMesh(self.dim)
        V3 = FunctionSpace(mesh3, "DG", 0)
        wid = TrialFunction(V3)
        uid = TestFunction(V3)
        self.diag_matrix = assemble(uid*wid*dx)
        self.diag_matrix.zero()

    def assemble_lumpedmm(self):
        """ Assembly lumped mass matrix - plays role of identity matrix
        """
        print("Assembling lumped mass matrix")
        mass_form = self.w * self.u * dx
        self.mass_matrix = assemble(mass_form)

        mass_action_form = action(mass_form, Constant(1))
        self.MM_terms = assemble(mass_action_form)

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
        self.explicit_matrices = np.empty(
            [self.csize_beta, self.csize_alpha], dtype=object)
        self.explicit_diffusion = np.empty(
            [self.csize_beta, self.csize_alpha], dtype=object)
        diag_vector = Vector(self.mesh.mpi_comm(), self.dim)
        global_min_timestep = float('inf')
        # Create explicit operator for each control in control set
        for ind_a, a in enumerate(self.ctrset_alpha):
            for ind_b, b in enumerate(self.ctrset_beta):
                if (ind_a*self.csize_beta + ind_b) % 10 == 0:
                    print(f'Assembling explicit matrix under control '
                          f'{ind_a*self.csize_beta + ind_b} out of '
                          f'{self.csize_alpha*self.csize_beta}')
                # coefficients in the interior
                advection_x = self.parameters.adv_x(a, b)
                advection_y = self.parameters.adv_y(a, b)
                reaction = self.parameters.lin(a, b)

                if t is not None:
                    advection_x.t = t
                    advection_y.t = t
                    reaction.t = t

                # Discretize PDE (set values on boundary rows to zero)
                b = np.array([advection_x, advection_y])
                E_int_form = (np.dot(b, grad(self.w)) +
                              reaction * self.w)*self.u * dx
                explicit_matrix = assemble(E_int_form)

                # Calculate nodewise minimum diffusion required to
                # impose monotonicity
                min_diffusion = [0]*explicit_matrix.size(0)
                for rows, row_num in getrows(explicit_matrix, self.laplacian,
                                             ignore=self.boundary_nodes_list):
                    min_diffusion[row_num] = calc_ad(rows, row_num)
                self.explicit_diffusion[ind_b][ind_a] = min_diffusion

                diffusion_matrix = apply_ad(self.laplacian, min_diffusion)
                explicit_matrix += diffusion_matrix

                explicit_matrix.get_diagonal(diag_vector)
                current_min_timestep = get_min_timestep(
                    diag_vector,
                    self.MM_terms,
                    self.boundary_nodes_list)

                global_min_timestep = min(
                    current_min_timestep, global_min_timestep)
                self.explicit_matrices[ind_b][ind_a] = toscipy(explicit_matrix)

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
        self.explicit_matrices = [[self.timestep_size * E -
                                   self.scipy_mass_matrix for E in Elist]
                                  for Elist in self.explicit_matrices]

        print('Checking if the explicit operators satisfy'
              ' monotonicity conditions')
        for matrix_list in self.explicit_matrices:
            for matrix in matrix_list:
                explicit_check(matrix, self.boundary_nodes_list)

    def assemble_RHS(self, t=None):
        """Assemble right hand side of the FBVP
        """
        print('Assembling RHS')
        self.forcing_terms = np.empty(
            [self.csize_beta, self.csize_alpha, self.dim])
        for ind_a, alpha in enumerate(self.ctrset_alpha):
            for ind_b, beta in enumerate(self.ctrset_beta):
                rhs = self.parameters.RHSt(alpha, beta)
                if t is not None:
                    rhs.t = t
                # Initialise forcing term
                F = np.array(assemble(rhs * self.u * dx)[:])
                self.forcing_terms[ind_b][ind_a] = self.timestep_size*F

    def assemble_HJBi(self, t=None):
        """ Assembly matrix discretizing second order terms after diffusion
        moved to the explicit operator is subtracted. Whenever amount of
        diffusion used to make some row of an explicit operator monotonic
        exceeds the amount of natural diffusion at that node we call it
        artificial diffusion. In such case, this row of the implicit operator
        is multiplied by zero
        """
        print('Assembling implicit operators')
        self.implicit_matrices = np.empty(
            [self.csize_beta, self.csize_alpha], dtype=object)
        remaining_diffusion = Function(self.V)
        max_art_dif = 0.0
        max_art_dif_loc = None
        diffusion_matrix = self.diag_matrix.copy()
        for ind_a, alpha in enumerate(self.ctrset_alpha):
            for ind_b, beta in enumerate(self.ctrset_beta):
                # TODO: Include assert making sure that diffusion is
                #  non-negative on all of the internal nodes
                diffusion = self.parameters.diffusion(alpha, beta)
                if t is not None:
                    diffusion.t = t
                diff = interpolate(diffusion, self.V).vector()
                if not np.all(diff >= 0):
                    raise Exception("Choose non-negative diffusion")

                diff_vec = np.array([
                    diff[i] if i not in self.boundary_nodes_list else 0.0
                    for i in range(self.dim)])

                artdif = self.explicit_diffusion[ind_b][ind_a] - diff_vec
                if np.amax(artdif) > max_art_dif:
                    max_art_dif = np.amax(artdif)
                    max_art_dif_loc = self.coords[np.argmax(artdif)]

                # discretise second order terms
                remaining_diffusion.vector(
                )[:] = np.maximum(-artdif, [0]*self.dim)
                diffusion_matrix.set_diagonal(remaining_diffusion.vector())

                Iab = matmult(diffusion_matrix, self.laplacian)
                self.implicit_matrices[ind_b][ind_a] = self.timestep_size * \
                    Iab + self.mass_matrix

        self.scipy_implicit_matrices = [[toscipy(mat) for mat in Ialist]
                                        for Ialist in self.implicit_matrices]
        print('Checking if the implicit operators satisfy'
              ' monotonicity conditions')
        for matrix_list in self.scipy_implicit_matrices:
            for matrix in matrix_list:
                implicit_check(matrix)

        with open(self.ad_data_path+'/ad.txt', 'a') as f:
            time_str = f' at time {t}\n' if t is not None else '\n'
            f.write(f'For mesh {self.mesh_name} max value of artificial'
                    ' diffusion coefficient was'
                    f' {max_art_dif} at {max_art_dif_loc}'
                    + time_str)


class Solver:
    def __init__(self, fbvp, howard_infsup=True, return_result=False,
                 output_mode=True, get_error=False, save_dir=None,
                 linear_mode=False, verbose=True, record_control=True):
        self.fbvp = fbvp
        self.solver = fbvp.parameters.solver
        self.howard_infsup = howard_infsup
        self.output_mode = output_mode
        self.get_error = get_error
        self.return_result = return_result
        self.linear_mode = linear_mode
        self.record_control = record_control
        if self.linear_mode:
            self.linear_control = [0, 0]
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

    def Howard_outer(self, beta):
        # v coresponds to v^{k+1}
        alpha = [0]*self.fbvp.dim
        ell = 0  # iteration counter
        while ell < self.fbvp.parameters.howmaxit:
            ell += 1
            how, alpha = self.Howard_inner(alpha, beta)

        # Create list of vectors (I*v^k+E*v^{k+1} -F) under different controls
        w = np.array(how[:])
        spv = np.array(self.v.vector()[:])
        Ev = [[Exp.dot(spv) for Exp in Exp_b_list]
              for Exp_b_list in self.fbvp.explicit_matrices]

        multlist = np.empty(
            [self.fbvp.csize_beta, self.fbvp.csize_alpha, self.fbvp.dim])
        for ind_b in range(self.fbvp.csize_beta):
            for ind_a in range(self.fbvp.csize_alpha):
                multlist[ind_b][ind_a] = \
                    (self.fbvp.scipy_implicit_matrices[ind_b][ind_a].dot(w) +
                     Ev[ind_b][ind_a] -
                     self.fbvp.forcing_terms[ind_b][ind_a])

        # Loop over vectors of values of (I*v^k+E*v^{k+1} -F) at each node and
        # record control which optimizes each of them
        if self.howard_infsup:
            next_ctr = np.argmin(np.max(multlist, axis=1), axis=0)
        else:
            next_ctr = np.argmax(np.min(multlist, axis=1), axis=0)
        return (how, next_ctr)

    def Howard_inner(self, alpha, beta):
        # v coresponds to v^{k+1}, save it to numpy array
        spv = np.array(self.v.vector()[:])

        Ev = {b: [Ea.dot(spv) for Ea in self.fbvp.explicit_matrices[b]]
              for b in set(beta)}
        # construct RHS under input control alfa
        rhs = self.v.vector().copy()
        rhs[:] = [self.fbvp.forcing_terms[beta[i]][a][i]
                  - Ev[beta[i]][a][i]
                  for i, a in enumerate(alpha)]

        # initialise vector with correct dimension to store solution
        how = self.v.vector().copy()

        # initalise matrix with a suitable sparsity pattern
        lhs = self.fbvp.implicit_matrices[0][0].copy()
        # construct implicit matrix under control (alfa, beta)
        for i, a in enumerate(alpha):
            lhs.set([self.fbvp.implicit_matrices[beta[i]][a].getrow(i)[1]],
                    [i], self.fbvp.implicit_matrices[beta[i]][a].getrow(i)[0])
        lhs.apply('insert')

        # solve linear problem to get next iterate of u
        for bc in self.fbvp.dirichlet_bcs:
            bc.apply(lhs, rhs)
        self.solver.solve(lhs, how, rhs)

        # Create list of vectors (I*v^k+E*v^{k+1} -F) under different controls
        spw = np.array(how[:])
        multlist = np.empty([self.fbvp.csize_alpha, self.fbvp.dim])
        Iw = {b: [Ia.dot(spw) for Ia in self.fbvp.scipy_implicit_matrices[b]]
              for b in set(beta)}

        for ind_a, _ in enumerate(self.fbvp.ctrset_alpha):
            for j in range(self.fbvp.dim):
                multlist[ind_a][j] = Iw[beta[j]][ind_a][j] \
                    + Ev[beta[j]][ind_a][j] - \
                    self.fbvp.forcing_terms[beta[j]][ind_a][j]

        # Loop over vectors of values of (I*v^k+E*v^{k+1} -F) at each node and
        # record control which optimizes each of them
        if self.howard_infsup:
            next_ctr = [np.argmax(vector) for vector in zip(*multlist)]
        else:
            next_ctr = [np.argmin(vector) for vector in zip(*multlist)]
        return (how, next_ctr)

    def time_iter(self):
        alpha = [0]*self.fbvp.dim  # initialize control
        for k in range(self.fbvp.timesteps - 1, -1, -1):  # go backwards in time
            time_start = time.perf_counter()
            t = k * self.fbvp.timestep_size  # Current time

            # update dirichlet boundary conditions if necessary
            if any(boundary_region in self.fbvp.parameters.time_dependent['boundary']
                   for boundary_region in self.fbvp.parameters.omegas):
                self.update_dirichlet_boundary(t)

            if self.fbvp.parameters.time_dependent['rhs']:
                self.fbvp.assemble_RHS(t)

            if self.fbvp.parameters.time_dependent['pde']:
                self.fbvp.assemble_HJBe(t+self.fbvp.timestep_size)
                self.fbvp.assemble_HJBi(t)

            if self.linear_mode:
                alpha = self.linear_control[0]
                beta = self.linear_control[1]
                A = self.fbvp.implicit_matrices[beta][alpha]
                b = self.v.vector().copy()
                b[:] = (self.fbvp.forcing_terms[beta][alpha] -
                        self.fbvp.explicit_matrices[beta][alpha]
                        .dot(np.array(self.v.vector()[:])))
                for bc in self.fbvp.dirichlet_bcs:
                    bc.apply(A, b)
                howit = self.v.vector().copy()
                self.solver.solve(A, howit, b)
            else:
                beta = [0]*self.fbvp.dim
                ell = 0  # iteration counter
                while ell < self.fbvp.parameters.howmaxit:
                    ell += 1
                    howit, beta = self.Howard_outer(beta)

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
            save_dir = f'out/{self.fbvp.parameters.experiment}/' \
                f'{self.fbvp.parameters.domain}/errors.json'
            error_calc(self.v,
                       self.fbvp.parameters.v_e,
                       self.fbvp.mesh,
                       self.fbvp.mesh_name,
                       save_dir=save_dir)

    def update_dirichlet_boundary(self, t):
        new_dirichlet_bcs = []
        for region in self.fbvp.parameters.omegas:
            if region in self.fbvp.parameters.time_dependent['boundary']:
                boundary_data = self.fbvp.parameters.RHS_bound[region]
                boundary_data.t = t
                new_dirichlet_bcs.append(
                    DirichletBC(self.fbvp.V, boundary_data,
                                self.fbvp.boundary_markers, region))
            else:
                new_dirichlet_bcs.append(
                    self.fbvp.dirichlet_bcs[region])
        self.fbvp.dirichlet_bcs = new_dirichlet_bcs
