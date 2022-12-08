using OrdinaryDiffEq
using Trixi


equations = Gaburro2D(1.0, 2.25*10^9, 1000, 0.0)

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
        alpha = 10^(-3)
    end
    
    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

initial_condition = initial_condition_test

  
volume_flux = (flux_central, flux_nonconservative_gaburro)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro),
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-3.0, -3.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 3.0,  3.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 0.0076)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

stepsize_callback = StepsizeCallback(cfl=0.01)

callbacks = CallbackSet(stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary