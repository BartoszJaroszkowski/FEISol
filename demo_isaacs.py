import os
import pathlib

from parameters.parameters_chase import Parameters
from PDE_Solver.isaacs import FBVP, Solver

EXPERIMENT_NAME = '2_player_chase'

os.chdir(pathlib.Path(__file__).parent.absolute())

domains = Parameters.domains
meshes = Parameters.meshes

if __name__ == "__main__":
    for domain in domains:
        for mesh in meshes:
            print(f'Calculation for mesh:{mesh}, domain: {domain} began')
            par = Parameters(domain, mesh, EXPERIMENT_NAME)
            fbvp = FBVP(mesh, par)
            solver = Solver(fbvp, output_mode=True, linear_mode=False,
                            howard_infsup=False, get_error=False)
            solver.time_iter()
