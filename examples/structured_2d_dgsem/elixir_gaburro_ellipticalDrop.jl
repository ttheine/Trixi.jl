using OrdinaryDiffEq
using Trixi


equations = Gaburro2D(1.0, 2.25*10^9, 1000, 9.81)

function initial_condition_test(x, t, equations::Gaburro2D)
  
    # liquid domain
    if((x[1]^2 + x[2]^2) <= 1)
        rho = 1000.0
        alpha = 1.0 - 10^(-3)
        v1 = -100.0 * x[1]
        v2 = 100.0 * x[2]
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        alpha = 0.0
    end
    
    return prim2cons(SVector(alpha, rho, v1, v2), equations)
end
  
initial_condition = initial_condition_test

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-3.0, -3.0)
coordinates_max = (3.0, 3.0)

cells_per_dimension = (2, 2)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 0.0076)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary