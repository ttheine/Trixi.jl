using OrdinaryDiffEq
using Revise
using Trixi
using Plots
using Printf

equations = Gaburro2D(1.0, 2.78*10^5, 1000.0, 9.81)

function initial_condition_const(x, t, equations::Gaburro2D)
    if(x[2] < 0.4)
        # liquid domain
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        alpha = 1.0 - 10^-3
    else
        # liquid domain
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        alpha = 10^-3
    end
    phi = x[2]
    
    return prim2cons(SVector(rho, v1, v2, alpha, phi), equations)
end

function initial_condition_line(x, t, equations::Gaburro2D)
  if((-x[1] + x[2]) <= 0.5)
      # liquid domain
      rho = 1000.0
      v1 = 0.0
      v2 = 0.0
      alpha = 1.0 - 10^-3
  else
      rho = 1000.0
      v1 = 0.0
      v2 = 0.0
      alpha = 10^-3
  end
  phi = x[2]
  
  return prim2cons(SVector(rho, v1, v2, alpha, phi), equations)
end

function initial_condition_exp(x, t, equations::Gaburro2D)
  if(x[2] < 0.4)
      # liquid domain
      rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) *(x[2] - 0.4))
      v1 = 0.0
      v2 = 0.0
      alpha = 1.0 - 10^-3
  else
      rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) *(x[2] - 1.0))
      v1 = 0.0
      v2 = 0.0
      alpha = 10^-3
  end
  phi = x[2]
  
  return prim2cons(SVector(rho, v1, v2, alpha, phi), equations)
end

function initial_condition_sin(x, t, equations::Gaburro2D)
  if(x[2] - 0.2*sin(x[1]) <= 0.0)
      # liquid domain
      rho = 1000.0
      v1 = 0.0
      v2 = 0.0
      alpha = 1.0 - 10^-3
  else
      rho = 1000.0
      v1 = 0.0
      v2 = 0.0
      alpha = 10^-3
  end
  phi = x[2]
  
  return prim2cons(SVector(rho, v1, v2, alpha, phi), equations)
end
  
initial_condition = initial_condition_line

boundary_conditions = (x_neg=boundary_condition_wall,
                       x_pos=boundary_condition_wall,
                       y_neg=boundary_condition_wall,
                       y_pos=boundary_condition_wall,)
  
volume_flux = (flux_central, flux_nonconservative_gaburro)
surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro)

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

coordinates_min = (-0.5, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 0.5, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=100_000, periodicity=(false,false))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, #source_terms=source_terms_gravity,
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

#amr_indicator = IndicatorHennemannGassner(semi,
 #                                         alpha_max=0.5,
  #                                        alpha_min=0.001,
   #                                       alpha_smooth=true,
    #                                      variable=first)

amr_indicator = IndicatorLöhner(semi, variable=alpha_rho)

#amr_controller = ControllerThreeLevel(semi, amr_indicator,
 #                                     base_level=4,
  #                                    med_level=5, med_threshold=0.1,
   #                                   max_level=6, max_threshold=0.6)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=2,
                                      max_level = 5, max_threshold=0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.4)

function save_my_plot(plot_data, variable_names;
  show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
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

  #pressure_matrix = equations.k0 .* plot_data.data[1]
  #pressure_matrix = pressure_matrix .- equations.k0
  #push!(plots, Plots.plot(heatmap(plot_data.x, plot_data.y, pressure_matrix), title = "pressure", width=10, height=10))

  # Create plot
  Plots.plot(plots...,)

  # Determine filename and save plot
  filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
  Plots.savefig(filename)
end

function save_my_plot_density(plot_data, variable_names;
  show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
  time=nothing, timestep=nothing)
  
  alpha_rho_data = plot_data["alpha_rho"]

  title = @sprintf("alpha_rho | 4th-order DG | t = %3.2f", time)
  
  Plots.plot(alpha_rho_data, 
             clim=(0.0,1200.0), 
             #colorbar_title="\ndensity",
             title=title,titlefontsize=9, 
             dpi=300,
             )

  Plots.plot!(getmesh(plot_data),linewidth=0.4)

  # Determine filename and save plot
  filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
  Plots.savefig(filename)
end

visualization_callback = VisualizationCallback(; interval=500,
                          solution_variables=cons2cons,
                          #variable_names=["rho"],
                          show_mesh=true,
                          #plot_data_creator=PlotData2D,
                          plot_creator=save_my_plot_density,
                          )

callbacks = CallbackSet(stepsize_callback, visualization_callback, alive_callback, amr_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
