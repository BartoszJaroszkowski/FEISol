from PDE_Solver.hjb_mixed import FBVP, Solver
from parameters.parameters_lshaped import Parameters
import pathlib
import os

os.chdir(pathlib.Path(__file__).parent.absolute())

EXPERIMENT_NAME = 'skorokhod'

if __name__ == '__main__':
    for domain in Parameters.domains:
        for mesh in Parameters.meshes:
            par = Parameters(domain, mesh, EXPERIMENT_NAME)
            fbvp = FBVP(mesh, par)
            solver = Solver(fbvp, howard_inf=False, get_error=False)
            solver.time_iter()
