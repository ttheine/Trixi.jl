using OrdinaryDiffEq
using Trixi
using Plots
using Printf

equations = Gaburro2D(1.0, 2.25*10^0, 1.0, 0.0)


function initial_condition_gauss_pulse(x, t, equations::Gaburro2D)
    
    if(((x[1]^2 + x[2]^2) <= 2))
        # liquid domain   
        rho = 1.0 * exp(-1/(1.0)^2*(x[1]^2 + x[2]^2)) + 0.1
        v1 = 0.0 * x[1]
        v2 = 0.0 * x[2]
        alpha = 1.0
    else
        rho = 1.0 * exp(-1/(1.0)^2*(x[1]^2 + x[2]^2)) + 0.1
        v1 = 0.0
        v2 = 0.0
        alpha = 0.9
    end

    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

function initial_condition_blast_wave(x, t, equations::Gaburro2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)
  
    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2  = r > 0.5 ? 0.0 : 0.1882 * sin_phi
    alpha = r > 0.5 ? 0.8 : 1.0
  
    return prim2cons(SVector(rho, v1, v2, alpha), equations)
  end
  
initial_condition = initial_condition_gauss_pulse


volume_flux = (flux_central, flux_nonconservative_gaburro)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro),
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-2.0, -2.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 2.0, 2.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)

boundary_conditions = BoundaryConditionDirichlet(initial_condition_gauss_pulse)                

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms = source_terms_gravity,
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 2.0)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.9)

function save_my_plot(plot_data, variable_names;
    show_mesh=false, plot_arguments=Dict{Symbol,Any}(),
    time=nothing, timestep=nothing)

  # Gather subplots
  plots = []
  for v in variable_names
    push!(plots, Plots.plot(plot_data[v]; plot_arguments...))
  end
  if show_mesh
    push!(plots, Plots.plot(getmesh(plot_data); plot_arguments...))
  end

  # Create plot
  Plots.plot(plots...,)

  # Determine filename and save plot
  filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
  Plots.savefig(filename)
end

#visualization_callback = VisualizationCallback(plot_creator=my_save_plot,interval=10, clims=(0,1.1), show_mesh=true)
visualization_callback = VisualizationCallback(; interval=50,
                            solution_variables=cons2prim,
                            #variable_names=["rho"],
                            show_mesh=false,
                            plot_data_creator=PlotData2D,
                            #plot_creator=save_my_plot,
                            )

callbacks = CallbackSet(alive_callback, visualization_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary