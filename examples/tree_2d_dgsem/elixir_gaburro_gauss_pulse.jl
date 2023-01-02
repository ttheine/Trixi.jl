using OrdinaryDiffEq
using Trixi
using Plots


equations = Gaburro2D(1.0, 2.25*10^4, 1000, 0.0)


function initial_condition_gauss(x, t, equations::Gaburro2D)
    
    if((x[1]^2 + x[2]^2) <= 1)
        # liquid domain   
        rho = 1000.0 * exp(-0.1*(x[1]^2 + x[2]^2))
        v1 = 10.0 * x[1]
        v2 = 10.0 * x[2]
        alpha = 1.0 - 10^(-3)
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        alpha = 10^(-3)
    end

    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end
  
initial_condition = initial_condition_gauss

  
volume_flux = (flux_central, flux_nonconservative_gaburro)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro),
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-3.0, -3.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 3.0, 3.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms = source_terms_gravity,
                                    boundary_conditions=boundary_condition_wall)

tspan = (0.0, 0.4)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.1)

function my_save_plot(plot_data, variable_names;
    show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
    time=nothing, timestep=nothing)
    
    Plots.plot(plot_data,clim=(0,1.1),title="Advected Blob");
    Plots.plot!(getmesh(plot_data))

    # Determine filename and save plot
    mkpath("out")
    #filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
    #Plots.savefig(filename)
end

#visualization_callback = VisualizationCallback(plot_creator=my_save_plot,interval=10, clims=(0,1.1), show_mesh=true)
visualization_callback = VisualizationCallback(; interval=500,
                            solution_variables=cons2prim,
                            #variable_names=["alpha_rho"],
                            show_mesh=false,
                            plot_data_creator=PlotData2D,
                            #plot_creator=show_plot,
                            )

callbacks = CallbackSet(alive_callback, visualization_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary