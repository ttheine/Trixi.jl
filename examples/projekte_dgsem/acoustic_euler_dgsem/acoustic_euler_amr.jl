using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the acoustic euler equations

equations = AcousticEulerEquations2D(rho=1.0, lambda=1.0)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=6, surface_flux=flux_lax_friedrichs)

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))


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

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

amr_indicator = IndicatorMax(semi, variable=pressure)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=3,
                                      med_level=4, med_threshold=2.00,
                                      max_level=5, max_threshold=2.75)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(amr_callback, stepsize_callback);


##############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-10, reltol=1.0e-10, dt=1.0, save_everystep=false, callback=callbacks);

#pd = PlotData2D(sol)
#plot(pd["p"])
#plot!(getmesh(pd))