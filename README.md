# FEISol - Finite Elements Isaacs Solver
## A Python library for fully nonlinear PDEs

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Mesh creation](#mesh-creation)
* [Parameters](#parameters)

## General info
This project is a demo for a fully nonlinear PDE solver based on the [FEniCS Project](https://fenicsproject.org/). It was designed for solving Isaacs problems
with general Dirichlet boundary conditions and Hamilton-Jacobi-Bellman problems with general mixed boundary conditions using FEM with P1 elements.
This solver is suitable for problems in non-divergence form, including those with degenerate diffusion and posed on nonconvex domains. It was created during
the work on the following articles regarding [Isaacs problem](), [HJB with mixed boundary conditions]() and [Heston model]().
	
## Setup
Note that since this project relies on FEniCS it can be only run on Ubuntu (this might be circumvented using Anaconda for Linux or Mac
, see [here](https://fenicsproject.org/download/) for details). To run this project, download it and simply run `setup.sh` script.
After that all demo files should be executable with python interpreter (tested on versions >=3.6). The results, stored in `out` folder
can be then visualised e.g. using [Paraview](https://www.paraview.org).

## Mesh creation
In order to solve own nonlinear problems one needs to first create own mesh using [Gmsh](https://gmsh.info). Note that requirement
of strict acuteness of the mesh makes it preferable to use Frontal-Delaunay algorithm for meshing as it usually yielded the best results in that regard.
In case mesh is not strictly acute, PDE solver will detect it and prompt the user. After creating mesh in `.msh` format one needs to transform it
into `.xdmf` format. In general, this can be done using [meshio](https://pypi.org/project/meshio/), but since FEniCS requires specific format
of the meshes author finds it more convenient (albeit slower) to use the following bash script for mesh conversion.

```
    #!/usr/bin/env bash
    ls -1 ${1}*.msh | xargs -n 1 bash -c 'dolfin-convert "$0" "${0%.*}.xml" &&
    meshio-convert "${0%.*}.xml" "${0%.*}.xdmf"'
    rm ${1}*.xml 
```

## Parameters
The second and final step to solving custom problems is creation of `Parameters` class derived from `PDESolver.tools.ParametersBase`.
The implemenatation may differ based on underlying problem but in general one needs to define coefficients of PDE operators
(both interior and boundary ones) and mark which of them are time-dependent. In case of mixed boundaries problems one also needs
to provide information in what kind of regions is the boundary divided. Available types are: Dirichlet, Robin(which includes Neumann)
and Robin with time derivative. See provided examples of `Parameters` classes for implementation details.
