using OrdinaryDiffEq
using Trixi
using Printf


equations = ThreeEquationModel2D(1.0, 8.78*10^5, 1000, 0.0)

function initial_condition_square(x, t, equations::ThreeEquationModel2D)
  
    # liquid domain
    if((abs(x[1]) <= 1) & (abs(x[2]) <= 1))
        rho = 1000.0
        alpha = 1.0 - 10^(-3)
        v1 = -2 * pi * x[2]
        v2 = 2 * pi * x[1]
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        alpha = 10^(-3)
    end
    
    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

initial_condition = initial_condition_square

  
volume_flux = (flux_central, flux_nonconservative_gaburro)
surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro)
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                           alpha_max=1.0,
                                           alpha_min=0.001,
                                           alpha_smooth=true,
                                           variable=density)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                   volume_flux_dg=volume_flux,
                                                   volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-5.0, -5.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 5.0,  5.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=500_000,
                periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_condition_wall)

tspan = (0.0, 1.0/(2*pi))
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

stepsize_callback = StepsizeCallback(cfl=0.4)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

function save_my_plot(plot_data, variable_names;
    show_mesh=false, plot_arguments=Dict{Symbol,Any}(),
    time=nothing, timestep=nothing)

  # Gather subplots
  plots = []
  for v in variable_names
    if v == "alpha_rho"
      push!(plots, Plots.plot(plot_data[v]; plot_arguments...))
    end
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

visualization_callback = VisualizationCallback(; interval=500,
                            solution_variables=cons2cons,
                            show_mesh=false,
                            plot_data_creator=PlotData2D,
                            plot_creator=save_my_plot,
                            )


callbacks = CallbackSet(stepsize_callback, alive_callback, visualization_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary