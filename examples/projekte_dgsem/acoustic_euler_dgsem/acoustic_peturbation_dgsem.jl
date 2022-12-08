using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the acoustic euler equations

equations = AcousticPerturbationEquations2D(v_mean_global=(0.0, 0.0), c_mean_global=1.0, rho_mean_global=1.0)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))


function initial_condition_test(x, t, equations::AcousticPerturbationEquations2D)
    v1 = 2 + sin(2* pi * x[1])
    v2 = 4 + cos(2* pi * x[2])
    p = 6 + sin(2 * pi * x[1]) + cos(2 * pi * x[2])
  
    return SVector(v1, v2, p, global_mean_vars(equations)...)
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

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)


# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.2)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, abstol=1.0e-8, reltol=1.0e-8,
        save_everystep=false, callback=callbacks);
# Print the timer summary
summary_callback()