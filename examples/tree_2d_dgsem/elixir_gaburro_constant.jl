using OrdinaryDiffEq
using Trixi


equations = Gaburro2D(1.0, 1.0*10^3, 1000.0, 9.81)

function initial_condition_const(x, t, equations::Gaburro2D)

    # liquid domain
    if x[2] < 0.5
        rho = 50.5
    else
        rho = 50.0
    end 
    rho = 100.0
    v1 = 0.0
    v2 = 0.0
    alpha = 1.0

    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

function initial_condition_const_exp(x, t, equations::Gaburro2D)

    # liquid domain
    rho = 100 * exp(- x[2])
    v1 = 0.0
    v2 = 0.0
    alpha = 1.0

    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end
  
initial_condition = initial_condition_const


boundary_conditions = (x_neg=boundary_condition_wall,
                       x_pos=boundary_condition_wall,
                       y_neg=boundary_condition_wall,
                       y_pos=boundary_condition_wall)
  
volume_flux = (flux_central, flux_nonconservative_gaburro)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro),
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-0.5, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 0.5, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=30_000,
                periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_gravity, boundary_conditions=boundary_conditions)

tspan = (0.0, 5.0)
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