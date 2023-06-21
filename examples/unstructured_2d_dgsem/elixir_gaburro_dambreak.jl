using OrdinaryDiffEq
using Revise
using Trixi
using Plots
using Printf

equations = Gaburro2D(1.0, 6.54*10^5, 1000.0, 9.81)

function initial_condition_dry_bed(x, t, equations::Gaburro2D)
  if((x[1] <= 0.0) && (x[2] <= 0.14618))
      # liquid domain
      rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) *(x[2] - 1.0))
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

function initial_condition_wet_bed(x, t, equations::Gaburro2D)
  if(((x[1] <= 0.0) && (x[2] <= 1.5)) )#|| ((x[1] >= 0.0) && (x[2] <= 0.75)))
      # liquid domain
      #rho = 1000.0
      rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) *(x[2] - 1.5))
      v1 = 0.0
      v2 = 0.0
      alpha = 1.0 - 10^-3
  elseif(((x[1] >= 0.0) && (x[2] <= 0.75)))
      # liquid domain
      #rho = 1000.0
      rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) *(x[2] - 0.75))
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
  
initial_condition = initial_condition_wet_bed

boundary_condition = Dict( :Bottom   => boundary_condition_wall,
                           :Right    => boundary_condition_wall,
                           :Top      => boundary_condition_wall,
                           :Left     => boundary_condition_wall);
  
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

###############################################################################
# Get the unstructured quad mesh from a file 
# create the unstructured mesh from your mesh file
mesh_file = joinpath("out", "tank_dambreak_big.mesh")

mesh = UnstructuredMesh2D(mesh_file)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms=source_terms_gravity,
                                    boundary_conditions=boundary_condition)

tspan = (0.0, 0.0)
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
             clim=(0.0,1200.0),#yticks = 0:0.1:4,
             #colorbar_title="\ndensity",
             c=:Blues,
             aspect_ratio = 10.0,
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
