import numpy as np
from petsc4py.PETSc import Mat
from scipy.sparse import csr_matrix
from scipy.stats import norm as normal
import os
import json
from dolfin import (as_backend_type, errornorm, norm, Vector, interpolate,
                    FunctionSpace, Matrix, PETScMatrix, XDMFFile, Function,
                    Mesh, MeshEditor, project, dof_to_vertex_map)
from abc import ABCMeta, abstractmethod


def getrows(*args, ignore=None):
    '''Generates rows of input matrices one-by-one. Works only for matrices of
     the same dimension, supports generic dolfin and PETSc formats.'''
    if ignore is None:
        ignore = set()
    if all(map(lambda x: isinstance(x, Matrix), args)):
        def getter(A, i): return A.getrow(i)
        dim = args[0].size(0)
    elif all(map(lambda x: isinstance(x, PETScMatrix)
                 or isinstance(x, Mat), args)):
        def getter(A, i): return A.getRow(i)
        dim = args[0].getSize()[0]
    else:
        raise Exception("Unsupported or non-matching matrix format."
                        "All matrices should be in the same format."
                        "Supported formats: Matrix, PETScMatrix")

    for i in range(dim):
        if i in ignore:
            continue
        if len(args) == 1:
            yield (getter(args[0], i), i)
        else:
            yield (map(getter, args, [i]*len(args)), i)


def l1_normalise(operator, l1_norm):

    if isinstance(operator, Vector):
        operator[:] = [operator[i] / l1_norm[i]
                       if l1_norm[i] else 0.0 for i in range(operator.size())]
        return

    for row, row_ind in getrows(operator):
        vals = []
        for pos in range(len(row[0])):
            vals.append(row[1][pos] / l1_norm[row_ind]
                        if l1_norm[row_ind] else 0.0)
        operator.set([vals], [row_ind], row[0])
        operator.apply('insert')


def set_zero_rows(mat, node_indices):
    for row_ind in node_indices:
        row = mat.getrow(row_ind)
        mat.set([[0.0]*len(row[1])], [row_ind], row[0])
        mat.apply('insert')


def calc_ad(rows, row_ind):
    ''' AD coefficient calculation - basically calculates minimal AD
    coefficient that will make all the off-diagonal terms non-positive'''
    ad_coeff = 0.0
    Erow, Lrow = rows
    for pos, column_ind in enumerate(Erow[0]):
        Lpos = np.where(Lrow[0] == column_ind)[0][0]
        if (Erow[1][pos] > 0.0 and column_ind != row_ind):
            if Lrow[1][Lpos] >= 0.0:
                raise ValueError("Non-negative off-diagonal term in the"
                                 f" Laplacian row:{row_ind} col:{column_ind}."
                                 " Make sure your mesh is strictly acute"
                                 )
            ad_temp = -(Erow[1][pos]) / (Lrow[1][Lpos])
            # Increase artificial diffusion coefficient value to counteract
            # rounding errors due to division
            ad_temp += 0.1*ad_temp
            ad_coeff = np.maximum(ad_coeff, ad_temp)
    return ad_coeff


def apply_ad(Lap, ratlist):
    artdif = Lap.copy()
    artdif.zero()
    artdif.apply('insert')
    for (Lrow, row_ind), rat in zip(getrows(Lap), ratlist):
        if rat > 0:
            artdif.set([rat * Lrow[1]], [row_ind], Lrow[0])
    artdif.apply('insert')
    return artdif


def matmult(A, B):
    ''' Matrix multiplication of dolfin matrices A and B where A is diagonal.
    The resulting matrix C is in PETSc format and needs to be converted to
    standard dolfin format.'''
    C = as_backend_type(A).mat().__mul__(as_backend_type(B).mat())
    C = todolfin(C, B)
    return C


def todolfin(A, B):
    '''Transform a matrix in PETSc format with the same sparsity pattern as
    dolfin matrix B into standard dolfin format'''
    A_d = B.copy()  # A_d is dolfin matrix
    for Arow, row_num in getrows(A):
        A_d.set([Arow[1]], [row_num], Arow[0])
    A_d.apply('insert')
    return A_d


def toscipy(A):
    '''Exports given dolfin matrix to scipy format'''
    mat = as_backend_type(A).mat()
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return csr


def explicit_check(E, ignore=None):
    """Checks if artificial diffusion was applied correctly to explicit operator
    """
    if ignore is not None:
        check_indices = list(set(range(E.shape[0])) - ignore)
        check = E[check_indices]
    else:
        check = E
    if np.any(check.data > 0):
        raise Exception("Positive Term In Explicit Operator")


def get_min_timestep(diag, MMdiag, ignore):
    timestep = float('inf')
    for i in range(diag.size()):
        if i in ignore or diag[i] <= 0:
            continue
        timestep = min(MMdiag[i]/diag[i], timestep)
    return timestep


def implicit_check(A):
    """Checks if implicit operator satisfies the monotonicity conditions:
        1. Sign condition
        2. Strict diagonal dominance

    Arguments:
        A {scipy.sparse.csr_matrix} -- implicit operator
        ignore {Iterable} -- iterable containing row indices to ignore

     Raises:
        Exception
    """
    nonzero = A.astype(bool).sum(axis=1)
    sumsigns = A.sign().sum(axis=1)
    diag_sign = np.sign(A.diagonal()).reshape((A.shape[0], 1))
    if np.any(sumsigns - diag_sign != -(nonzero-1)):
        row = np.where(sumsigns - diag_sign != -(nonzero-1))[0][0]
        raise Exception(
            f'Wrong sign in impicit matrix row: {row}'
        )

    D = np.abs(A.diagonal())
    D = D.reshape(-1, 1)
    S = np.sum(np.abs(A), axis=1) - D
    if not np.all(D > S):
        row = np.where(D <= S)[0][0]
        raise Exception(f'Matrix is not diagonally dominant - see row: {row}')


def error_calc(v, v_e, mesh, mesh_name, save_dir='errors.json'):
    try:
        with open(save_dir, 'r') as f:
            errors = json.load(f)
    except FileNotFoundError:
        errors = {}
    new_errors = {}
    v_e.t = 0.0
    v_int = interpolate(v_e, FunctionSpace(mesh, 'CG', 1))
    new_errors['Linf'] = norm(v.vector() - v_int.vector(), 'linf')
    new_errors['L2'] = errornorm(v_e, v, mesh=mesh, degree_rise=3)
    new_errors['H1'] = errornorm(
        v_e, v, norm_type='H1', mesh=mesh, degree_rise=3)
    errors[mesh_name] = new_errors
    with open(save_dir, 'w') as f:
        json.dump(errors, f)


def get_BS(dofmap, bnodes, mesh, K, r, rho, xi, T, t):
    """ Calculates exact value of Black scholes equation on boundary nodes
     of a given mesh """
    value_vector = np.zeros(len(mesh.coordinates()))
    # dofmap is dof_to_vertex_map(FunctionSpace)
    for index, point in enumerate(mesh.coordinates()[dofmap]):
        if index not in bnodes:
            continue
        y, z = point[0], point[1]
        S = np.exp(y + rho/np.sqrt(1-rho**2)*z)
        sigma = np.sqrt(xi/np.sqrt(1-rho**2)*z)
        tau = T - t  # time to maturity
        S, sigma, tau = [1E-8 if var == 0 else var for var in (S, sigma, tau)]
        d1 = (np.log(S/K) + (r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
        d2 = (np.log(S/K) + (r-0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
        V_C = S*normal.cdf(d1)-K*np.exp(-r*tau)*normal.cdf(d2)
        value_vector[index] = V_C
    return value_vector


class ParametersBase(metaclass=ABCMeta):
    def __init__(self, domain, mesh_name, experiment):
        if not self.domains or not self.meshes:
            raise Exception("You need to provide at least one domain and mesh"
                            "and name of the experiment")
        self.domain = domain
        self.mesh_name = mesh_name
        self.set_coefficients()
        self.experiment = experiment

    @abstractmethod
    def set_coefficients(self):
        """Define coefficients of PDE and set parameters of
        numerical experiment
        """
        pass

    @abstractmethod
    def set_boundary_conditions(self, mesh):
        """ Given mesh, define subdomains and boundary regions that will be
            used to calculate boundary conditions.
        """
        pass

    def calculate_save_interval(self):
        self.save_interval = 1

    def get_number_of_timesteps(self):
        timestep_data = f'meshes/{self.domain}/Mvalues-{self.experiment}.json'
        try:
            with open(timestep_data) as f:
                min_timesteps = json.load(f)
                self.timesteps = min_timesteps[self.mesh_name]
        except (FileNotFoundError, KeyError):
            print('Warning: timestep data not found, setting M to None')
            self.timesteps = None
        return self.timesteps

    def get_mesh_path(self):
        return f'meshes/{self.domain}/{self.domain}_{self.mesh_name}.xdmf'


def heston_transform(parameters, get_error=False, file_path=None,
                     plot_deltas=False):
    experiment = parameters.experiment
    domain_name = parameters.domain
    mesh_name = parameters.mesh

    mesh = Mesh()
    mesh_path = parameters.get_mesh_path()
    with XDMFFile(mesh_path) as f:
        f.read(mesh)

    # Create functions in function spaces defined by both original and
    # transformed meshes and copy value function data from one to another
    t_mesh_path = f'meshes/{domain_name}/t_{domain_name}_{mesh_name}.xdmf'
    if not os.path.exists(t_mesh_path):
        mesh2 = get_original_mesh(t_mesh_path, mesh, parameters)
    else:
        mesh2 = Mesh()
        with XDMFFile(t_mesh_path) as f:
            f.read(mesh2)

    V = FunctionSpace(mesh, 'CG', 1)
    t_V = FunctionSpace(mesh2, 'CG', 1)
    value = Function(V)
    control = Function(V)
    t_value = Function(t_V)
    t_control = Function(t_V)
    if not file_path:
        f = XDMFFile(f'out/{experiment}/{domain_name}/{mesh_name}/v.xdmf')
        t_f = XDMFFile(f'out/{experiment}/{domain_name}/{mesh_name}/t_v.xdmf')
    else:
        f = XDMFFile(file_path)
        path, file_name = os.path.split(file_path)
        t_f = XDMFFile(os.path.join(path, f't_{file_name}'))
    # Save Final Time Condition
    f.read_checkpoint(value, 'value_func', 0)
    t_value.vector()[:] = value.vector()[:]
    t_f.write_checkpoint(t_value, 'value_func', parameters.T)

    # Iterate through remaining timesteps (only some of them are
    #  saved to the file)
    i = 1
    parameters.calculate_save_interval()
    dt = parameters.T / parameters.M
    for k in range(parameters.M-1, -1, -1):
        if k % parameters.save_interval != 0:
            continue
        f.read_checkpoint(value, 'value_func', i)
        f.read_checkpoint(control, 'control', i-1)
        t_value.vector()[:] = value.vector()[:]
        t_control.vector()[:] = control.vector()[:]
        t_f.write_checkpoint(t_value, 'value_func', k * dt,
                             XDMFFile.Encoding.HDF5, True)
        t_f.write_checkpoint(t_control, 'control', k * dt,
                             XDMFFile.Encoding.HDF5, True)
        if plot_deltas:
            delta_S = project(t_value.dx(0), t_V)
            t_f.write_checkpoint(delta_S, 'delta_S', k * dt,
                                 XDMFFile.Encoding.HDF5, True)
            delta_v = project(t_value.dx(1), t_V)
            t_f.write_checkpoint(delta_v, 'delta_v', k * dt,
                                 XDMFFile.Encoding.HDF5, True)
        i += 1

        if k == 0 and get_error:
            error_calc(t_value, parameters.t_v_e, mesh2,
                       f't_{mesh_name}',
                       f'out/{experiment}/{domain_name}/errors.json')


def get_original_mesh(t_mesh_path, mesh, par):
    def transform(coordinates, xi, rho):
        S = np.exp(coordinates[:, 0]+coordinates[:, 1]*rho/np.sqrt(1-rho**2))
        v = 50*coordinates[:, 1]*xi/np.sqrt(1-rho**2)
        new_coords = np.column_stack((S, v))
        return new_coords

    if os.path.exists(t_mesh_path):
        t_mesh = Mesh()
        with XDMFFile(t_mesh_path) as f:
            f.read(t_mesh)
        return t_mesh

    # Construct the mesh under transformed variables
    new_coords = transform(mesh.coordinates(), xi=par.xi, rho=par.rho)

    t_mesh = Mesh()
    editor = MeshEditor()
    editor.open(t_mesh, 'triangle', 2, 2)
    editor.init_vertices(len(new_coords))
    editor.init_cells(len(mesh.cells()))

    for i, vertex in enumerate(new_coords):
        editor.add_vertex(i, vertex)
    for i, cell in enumerate(mesh.cells()):
        editor.add_cell(i, cell)
    editor.close()

    with XDMFFile(t_mesh_path) as f:
        f.write(t_mesh)
    return t_mesh


def plot_deltas_2D(vertex_vector, mesh_path, save_dir):
    mesh = Mesh()
    with XDMFFile(mesh_path) as f:
        f.read(mesh)
    V = FunctionSpace(mesh, 'CG', 1)
    value = Function(V)
    value.vector()[:] = vertex_vector[dof_to_vertex_map(V)]
    delta_S = project(value.dx(0), V)
    delta_file = XDMFFile(save_dir)
    delta_file.write_checkpoint(delta_S, 'delta_S', 0,
                                XDMFFile.Encoding.HDF5, True)
    delta_v = project(value.dx(1), V)
    delta_file.write_checkpoint(delta_v, 'delta_v', 0,
                                XDMFFile.Encoding.HDF5, True)


def calc_value_differences(file_path1, file_path2, output_path,
                           func_space, t_func_space, parameters):
    f1 = XDMFFile(file_path1)
    f2 = XDMFFile(file_path2)
    f_out = XDMFFile(output_path)

    value_func1 = Function(func_space)
    value_func2 = Function(func_space)
    result = Function(t_func_space)

    f1.read_checkpoint(value_func1, 'value_func', 0)
    f2.read_checkpoint(value_func2, 'value_func', 0)
    result.vector()[:] \
        = value_func1.vector()[:] - value_func2.vector()[:]
    f_out.write_checkpoint(result, 'diff', parameters.T,
                           XDMFFile.Encoding.HDF5, True)

    i = 1
    dt = parameters.T / parameters.M
    for k in range(parameters.M-1, -1, -1):
        if k % parameters.save_interval != 0:
            continue
        f1.read_checkpoint(value_func1, 'value_func', i)
        f2.read_checkpoint(value_func2, 'value_func', i)
        result.vector()[:] \
            = value_func1.vector()[:] - value_func2.vector()[:]
        f_out.write_checkpoint(result, 'diff', k*dt,
                               XDMFFile.Encoding.HDF5, True)
        i += 1
