using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the acoustic euler equations

equations = AcousticEulerEquations2D(rho=1.0, lambda=1.0)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))


function initial_condition_test(x, t, equations::AcousticEulerEquations2D)
    v1 = 2 + sin(2* pi * x[1])
    v2 = 4 + cos(2* pi * x[2])
    p = 6 + sin(2 * pi * x[1]) + cos(2 * pi * x[2])

    return SVector(v1, v2, p)
end

initial_condition = initial_condition_test

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)


stepsize_callback = StepsizeCallback(cfl=0.9)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(stepsize_callback)


##############################################################################
# run the simulation

sol_euler = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-10, reltol=1.0e-10, dt=1.0, save_everystep=false, callback=callbacks);
sol_euler_final = sol_euler[2]


##########################################################################################################################################
# Acoustic perturbation


equations = AcousticPerturbationEquations2D(v_mean_global=(0.0, 0.0), c_mean_global=1.0, rho_mean_global=1.0)


function initial_condition_test(x, t, equations::AcousticPerturbationEquations2D)
    v1 = 2 + sin(2* pi * x[1])
    v2 = 4 + cos(2* pi * x[2])
    p = 6 + sin(2 * pi * x[1]) + cos(2 * pi * x[2])

    prim = SVector(v1, v2, p, global_mean_vars(equations)...)

  return prim2cons(prim, equations)
end

initial_condition = initial_condition_test

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)


stepsize_callback = StepsizeCallback(cfl=0.2)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(stepsize_callback)


##############################################################################
# run the simulation

sol_peturbation_all_variables = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-10, reltol=1.0e-10, dt=1.0, save_everystep=false, callback=callbacks);
sol_peturbation_all_variables_final = sol_peturbation_all_variables[2]

error = zeros((4 * 2^4)^2 * 3)

for i = 1:(4 * 2^4)^2
  error[i] = abs(sol_euler_final[(i-1)*3 + 3] - sol_peturbation_all_variables_final[(i-1)*7 + 3])
end

println("Max. Fehler beim Druck p zwischen den beiden Equations:  ", norm(error, Inf))

#CFL = 0.1, Gitter: 16 x 16 -> Fehler: 2.7748470188271313e-11
#CFL = 0.2, Gitter: 32 x 32 -> Fehler: 2.396927101244728e-11
#CFL = 0.2, Gitter: 64 x 64 -> Fehler: 5.302425165609748e-13

#CFL = 0.1, Gitter: 32 x 32 -> Fehler: 9.903189379656396e-13