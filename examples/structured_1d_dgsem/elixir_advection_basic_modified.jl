using Trixi, OrdinaryDiffEq, Plots, BenchmarkTools

# equation with a advection_velocity of `2`.
advection_velocity = 2.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# create DG solver with flux lax friedrichs and LGL basis
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# distretize domain with `TreeMesh` (works with hypercubes)
coordinates_min = 0.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2, # number of elements = 2^level, 
                n_cells_max=30_000)

# create initial condition and semidiscretization
initial_condition_sine_wave(x, t, equations) = SVector(sin(2 * pi * sum(x)))
#initial_condition_sine_wave(x, t, equations) = SVector(1)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

# solve
tspan = (0.0, 10.0)
ode_trixi  = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy, energy_total))

# The AliveCallback prints short status information in regular intervals
#alive_callback = AliveCallback(analysis_interval=analysis_interval)

# The SaveRestartCallback allows to save a file from which a Trixi simulation can be restarted
save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.02)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback,
                        analysis_callback, #alive_callback,
                        save_restart, save_solution,
                        stepsize_callback)


#@btime sol_trixi  = solve(ode_trixi, RDPK3SpFSAL49(), abstol=1.0e-10, reltol=1.0e-10, save_everystep=false);
sol  = solve(ode_trixi, RDPK3SpFSAL49(), dt=1.0, abstol=1.0e-10, reltol=1.0e-10, save_everystep=false, callback = callbacks);

# Print the timer summary
summary_callback()