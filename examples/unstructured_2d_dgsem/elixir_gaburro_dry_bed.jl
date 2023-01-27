using OrdinaryDiffEq
using Trixi
using Printf


equations = Gaburro2D(1.0, 6.37*10^5, 1000, 9.81)

function initial_condition_dry(x, t, equations::Gaburro2D)
  
    # liquid domain
    if((-5.0 <= x[1] <= 0.0) & (0.0 <= x[2] <= 1.4618))
        rho = equations.rho_0 * exp(-(equations.gravity*equations.rho_0/equations.k0) * (x[2] - 1.5))
        alpha = 1.0 - 10^(-3)
        v1 = 0.0
        v2 = 0.0
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        alpha = 10^(-3)
    end
    
    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

initial_condition = initial_condition_dry

  
volume_flux = (flux_central, flux_nonconservative_gaburro)
surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro)

basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=density)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                  volume_flux_dg=volume_flux,
                                                  volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

boundary_condition = Dict( :Bottom  => boundary_condition_wall,
                            :Right  => boundary_condition_wall,
                            :Top  => boundary_condition_wall,
                            :Left  => boundary_condition_wall)

###############################################################################
# Get the unstructured quad mesh from a file 
# create the unstructured mesh from your mesh file
mesh_file = joinpath("out", "tank.mesh")

mesh = UnstructuredMesh2D(mesh_file)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, 
                                    source_terms=source_terms_gravity, boundary_conditions=boundary_condition)

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.1)

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

  pressure_matrix = equations.k0 .* plot_data.data[1]
  pressure_matrix = pressure_matrix .- equations.k0
  push!(plots, Plots.plot(heatmap(plot_data.x, plot_data.y, pressure_matrix), title = "pressure", width=10, height=10))

  # Create plot
  Plots.plot(plots...,)

  # Determine filename and save plot
  filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
  Plots.savefig(filename)
end

visualization_callback = VisualizationCallback(; interval=1000,
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