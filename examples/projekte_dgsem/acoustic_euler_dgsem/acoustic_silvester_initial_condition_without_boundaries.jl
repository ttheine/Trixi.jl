using OrdinaryDiffEq
using Trixi
using LinearAlgebra


###############################################################################
# simulation of a new year's firecracker between two houses


equations = AcousticEulerEquations2D(rho=1.0, lambda=1.0)

function initial_condition_silvester(x, t, equations::AcousticEulerEquations2D)
    v1 = 0.0

    if (0.4 <= x[1] <= 0.6) && (0.1 <= x[2] <= 0.4) 
        v2 = 1.0
    else
        v2 = 0.0
    end

    if (0.4 <= x[1] <= 0.6) & (0.1 <= x[2] <= 0.4)
        p = 3 * exp(-((x[1] - 0.5)^2 + (x[2] - 0.25)^2) / (2 * 10^(-2))) + 2
    else
        p = 2.0
    end

    return SVector(v1, v2, p)
end

initial_condition = initial_condition_silvester


# Create DG solver with polynomial degree and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=6, surface_flux=flux_lax_friedrichs)

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=30_000) # set maximum capacity of tree data structure


# Create semidiscretization with all spatial discretization-related components
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.5
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)


stepsize_callback = StepsizeCallback(cfl=0.001)  
# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(stepsize_callback)


##############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-10, reltol=1.0e-10, dt=1.0, save_everystep=false, callback=callbacks);
sol_euler = sol[2]



######################################################################################################################################################
# Acoustic perturbation


equations = AcousticPerturbationEquations2D(v_mean_global=(0.0, 0.0), c_mean_global=1.0, rho_mean_global=1.0)

function initial_condition_silvester(x, t, equations::AcousticPerturbationEquations2D)
    v1 = 0.0

    if (0.4 <= x[1] <= 0.6) && (0.1 <= x[2] <= 0.4) 
        v2 = 1.0
    else
        v2 = 0.0
    end

    if (0.4 <= x[1] <= 0.6) & (0.1 <= x[2] <= 0.4)
        p = 3 * exp(-((x[1] - 0.5)^2 + (x[2] - 0.25)^2) / (2 * 10^(-2))) + 2
    else
        p = 2.0
    end

    return SVector(v1, v2, p, global_mean_vars(equations)...)
end

initial_condition = initial_condition_silvester


# Create DG solver with polynomial degree and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=6, surface_flux=flux_lax_friedrichs)

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=30_000) # set maximum capacity of tree data structure


# Create semidiscretization with all spatial discretization-related components
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.5
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)


stepsize_callback = StepsizeCallback(cfl=0.001)                                     
# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(stepsize_callback)


##############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-10, reltol=1.0e-10, dt=1.0, save_everystep=false, callback=callbacks);
sol_peturbation = sol[2]

error = zeros(50176)

for i = 1:50176
  error[i] = abs(sol_euler[(i-1)*3 + 3] - sol_peturbation[(i-1)*7 + 3])
end

println("Max. Fehler beim Druck p zwischen den beiden Equations:  ", norm(error, Inf))

#polydeg = 6:
#CFL = 0.1, Gitter 16 x 16, Fehler: 1.2522423491478918e-6
#CFL = 0.05, Gitter 16 x 16, Fehler: 8.28279045350655e-8
#CFL = 0.01, Gitter 16 x 16, Fehler: 1.3817484934008917e-9

#CFL = 0.1, Gitter 32 x 32, Fehler: 6.640059233564699e-7
#CFL = 0.05, Gitter 32 x 32, Fehler: 4.9570883842164903e-8
#CFL = 0.01, Gitter 32 x 32, Fehler: 
