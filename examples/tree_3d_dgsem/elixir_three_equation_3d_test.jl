using OrdinaryDiffEq
using Revise
using Trixi
using Plots
using Printf

equations = ThreeEquationModel3D(1.0, 2.62*10^5, 1000.0, 9.81)


function initial_condition_gauss_bell(x, t, equations::ThreeEquationModel3D)
    
    if((exp(-x[1]^2 - x[2]^2) >= x[3]))
        # liquid domain   
        rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) * (x[3] - exp(-x[1]^2 - x[2]^2)))
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 1.0 - 10^(-3)
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 10^(-3)
    end
    phi = x[3]
    return prim2cons(SVector(rho, v1, v2, v3, alpha, phi), equations)
end

function initial_condition_drop(x, t, equations::ThreeEquationModel3D)
    
    if(x[3] <= 0.25)
        # liquid domain   
        rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) * (x[3] - 0.25))
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 1.0 - 10^(-3)
    elseif((x[1]^2 + x[2]^2 + (x[3]-0.6)^2 <= 0.04))
        # liquid domain
        rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) * (x[3] - sqrt(0.04 - x[1]^2 -x[2]^2) - 0.6))
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 1.0 - 10^(-3)
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 10^(-3)
    end
    phi = x[3]
    return prim2cons(SVector(rho, v1, v2, v3, alpha, phi), equations)
end

function initial_condition_dambreak(x, t, equations::ThreeEquationModel3D)
    
    if((x[3] <= 1.5) && (x[2] <= 0))
        # liquid domain   
        rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) * (x[3] - 1.5))
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 1.0 - 10^(-3)
    elseif((x[3] <= 0.75) && (x[2] > 0))
        # liquid domain
        rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) * (x[3] - 0.75))
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 1.0 - 10^(-3)
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 10^(-3)
    end
    phi = x[3]
    return prim2cons(SVector(rho, v1, v2, v3, alpha, phi), equations)
end

function initial_condition_dambreak_wall(x, t, equations::ThreeEquationModel3D)
    
    if((x[3] <= 0.6) && (x[2] <= 1.2))
        # liquid domain   
        rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) * (x[3] - 0.6))
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 1.0 - 10^(-3)
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        alpha = 10^(-3)
    end
    phi = x[3]
    return prim2cons(SVector(rho, v1, v2, v3, alpha, phi), equations)
end

  
initial_condition = initial_condition_dambreak_wall


volume_flux = (flux_central, flux_nonconservative_gaburro_well)
surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro_well)
#solver = DGSEM(polydeg=3, surface_flux=surface_flux,
 #                volume_integral=VolumeIntegralFluxDifferencing(volume_flux))


basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                          alpha_max=1.0,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=alpha_rho)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                  volume_flux_dg=volume_flux,
                                                  volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (3.2, 3.2, 1.8) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with non-periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=1_000_000,
                periodicity=false)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, #source_terms = source_terms_gravity,
                                    boundary_conditions=boundary_condition_wall)

tspan = (0.0, 1.7)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=500, save_initial_solution=true,save_final_solution=true, solution_variables=cons2cons)

stepsize_callback = StepsizeCallback(cfl=0.4)


visualization_callback = VisualizationCallback(; interval=100,
                            solution_variables=cons2prim,
                            #variable_names=["rho"],
                            show_mesh=false,
                            plot_data_creator=PlotData2D,
                            #plot_creator=save_my_plot,
                            )

callbacks = CallbackSet(alive_callback, stepsize_callback, save_solution)#visualization_callback

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary