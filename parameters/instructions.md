# PARAMETERS CLASS CREATION


## TO BE DEFINED AS CLASS VARIABLES

List of mesh file names contained in folder names stored in domains.
```
domains = list[str]
meshes = list[str]
```


## TO DE DEFINED INSIDE SET_COEFFICIENTS()

Final time
```
T = int
```

---------------------------------
### Parameters needed for creation of control sets

MixedBC:
```
alpha_range = list[int] or tuple[int] (of size 2)
control_set_size = int
```

Isaacs:
```
alpha_range = list[int] or tuple[int] (of size 2)
alpha_size = int

beta_range = list[int] or tuple[int] (of size 2)
beta_size = int
```
----------------------------------
Maximum number of iterations of Howard's algorithm which solves nonlinear problem
```
howmaxit = int
```
Solver used by FEniCS, see more [here](https://fenicsproject.org/pub/tutorial/html/._ftut1017.html)
```
solver = GenericLinearSolver
```
### PDE coefficients
```
diffusion = Callable[[int],Expression]
```
Advection in x and y direction
```
adv_x = Callable[[int],Expression]
adv_y = Callable[[int],Expression]
```
Reaction term
```
lin = Callable[[int],Expression]
```
Forcing term (right hand side of the PDE)
```
RHSt = Callable[[int],Expression]
```
Final time condition
```
ft = Expression
```

## TO BE DEFINED INSIDE SET_BOUNDARY_CONDITIONS()

Dictionary of boundary regions
```
omegas = dict[int, Subdomain]
```
---------------------------
For mixedBC describe what types of boundary conditions are imposed in each region
```
regions = dict['Dirichlet': List[int], 'Robin': List[int], 'RobinTime': list[int]]
```
---------------------------
Forcing term for each boundary region
```
RHS_bound = dict[int, Expression]
```
Denote if PDE coefficients or forcing term are time dependent + list all time dependent boundary regions(only Dirichlet boundary type supported)
```
time_dependent = dict['pde': boolean, 'rhs': boolean, 'boundary': list[int]]
```
