using Trixi
using Plots
using Printf
using OrdinaryDiffEq

equations = ThreeEquationModel2D(1.0, 2.78*10^5, 1000.0, 0.0)

function initial_condition_water_jet(x, t, equations::ThreeEquationModel2D)
  if((x[2] - 1.6*x[1] <= 0.0) && (x[2] - 1.6 * x[1] >= -1.6))
      # liquid domain
      rho = 1000.0
      v1 = 5 * -0.52999894
      v2 = 5 * -0.847998304
      alpha = 1.0 - 10^-3
  else
      rho = 1000.0
      v1 = 0.0
      v2 = 0.0
      alpha = 10^-3
  end
  
  return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

function initial_condition_water_jet_vertical(x, t, equations::ThreeEquationModel2D)
  if(-0.5 <= x[1] <= 0.5)
      # liquid domain
      rho = 1000.0
      v1 = 5 * -0.0
      v2 = 5 * -1.0
      alpha = 1.0 - 10^-3
  else
      rho = 1000.0
      v1 = 0.0
      v2 = 0.0
      alpha = 10^-3
  end
  
  return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

initial_condition = initial_condition_water_jet_vertical

boundary_conditions = (x_neg=boundary_condition_wall,
                       x_pos=boundary_condition_wall,
                       y_neg=boundary_condition_wall,
                       y_pos=BoundaryConditionDirichlet(initial_condition),)
  
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

coordinates_min = (-5.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 5.0, 10.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=8,
                n_cells_max=100_000, periodicity=(false,false))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, 
                source_terms=source_terms_gravity, boundary_conditions=boundary_conditions)

tspan = (0.0, 3.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.8)


function save_my_plot_density(plot_data, variable_names;
  show_mesh=false, plot_arguments=Dict{Symbol,Any}(),
  time=nothing, timestep=nothing)
  
  alpha_rho_data = plot_data["alpha_rho"]

  title = @sprintf("alpha_rho | 4th-order DG | t = %3.2f", time)
  
  Plots.plot(alpha_rho_data, 
             clim=(0.0,1000.0), 
             #colorbar_title="\ndensity",
             title=title,titlefontsize=9, 
             dpi=300,
             )

  #Plots.plot!(getmesh(plot_data),linewidth=0.4)

  # Determine filename and save plot
  filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
  Plots.savefig(filename)
end

visualization_callback = VisualizationCallback(; interval=500,
                          solution_variables=cons2cons,
                          #variable_names=["rho"],
                          show_mesh=false,
                          #plot_data_creator=PlotData2D,
                          plot_creator=save_my_plot_density,
                          )

callbacks = CallbackSet(stepsize_callback, visualization_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
