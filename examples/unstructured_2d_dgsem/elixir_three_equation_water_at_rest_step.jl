using OrdinaryDiffEq
using Revise
using Trixi
using Plots
using Printf
using LinearAlgebra

equations = ThreeEquationModel2D(1.0, 2.25*10^9, 1000.0, 9.81)

function initial_condition_rest(x, t, equations::ThreeEquationModel2D)
    # liquid domain
    rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) *(x[2] - 1.0))
    v1 = 0.0
    v2 = 0.0
    alpha = 1.0
    phi = x[2]
    
    return prim2cons(SVector(rho, v1, v2, alpha, phi), equations)
end
  
initial_condition = initial_condition_rest

boundary_condition = Dict( :Bottom1  => boundary_condition_wall,
                            :StepLeft  => boundary_condition_wall,
                            :StepTop => boundary_condition_wall,
                            :StepRight  => boundary_condition_wall,
                            :Bottom2  => boundary_condition_wall,
                            :Right  => boundary_condition_wall,
                            :Top  => boundary_condition_wall,
                            :Left  => boundary_condition_wall)

###############################################################################
# Get the DG approximation space

volume_flux = (flux_central, flux_nonconservative_gaburro_well)
surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro_well)

solver = DGSEM(polydeg=3, surface_flux=surface_flux,
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the unstructured quad mesh from a file 
# create the unstructured mesh from your mesh file
mesh_file = joinpath("out", "tank_water_at_rest_step.inp")

#mesh = UnstructuredMesh2D(mesh_file, periodicity=false)
mesh = P4estMesh{2}(mesh_file, polydeg=1, initial_refinement_level=0)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, #source_terms = source_terms_gravity,
                                    boundary_conditions=boundary_condition)

###############################################################################
# ODE solvers, callbacks, etc.

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.2)

function save_my_plot_density(plot_data, variable_names;
                              show_mesh=false, plot_arguments=Dict{Symbol,Any}(),
                              time=nothing, timestep=nothing)
    
    alpha_rho_data = plot_data["alpha_rho"]
  
    title = @sprintf("alpha_rho | 4th order DG | t = %3.2f", time)
    
    Plots.plot(alpha_rho_data, 
               clim=(1000.0,1035.0), 
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

time_series = TimeSeriesCallback(semi, [(0.0, 0.3)];
                                 interval=5,
                                 solution_variables=cons2cons,
                                 filename="tseries.h5")

callbacks = CallbackSet(stepsize_callback, alive_callback, visualization_callback, time_series)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54( williamson_condition=false), maxiters=1e7,
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
