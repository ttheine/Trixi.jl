using OrdinaryDiffEq
using Revise
using Trixi
using Plots
using Printf
using LinearAlgebra


equations = Gaburro2D(1.0, 2.78*10^5, 1000.0, 9.81)

function initial_condition_rest(x, t, equations::Gaburro2D)
    # liquid domain
    rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) *(x[2] - 1.0))
    v1 = 0.0
    v2 = 0.0
    alpha = 1.0
    phi = x[2]
    
    return prim2cons(SVector(rho, v1, v2, alpha, phi), equations)
end
  
initial_condition = initial_condition_rest

boundary_conditions = (x_neg=boundary_condition_wall,
                       x_pos=boundary_condition_wall,
                       y_neg=boundary_condition_wall,
                       y_pos=boundary_condition_wall,)
  
volume_flux = (flux_central, flux_nonconservative_gaburro_well)
surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro_well)

solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro_well),
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

basis = LobattoLegendreBasis(3)
#indicator_sc = IndicatorHennemannGassner(equations, basis,
 #                                         alpha_max=0.5,
  #                                        alpha_min=0.001,
   #                                       alpha_smooth=true,
    #                                      variable=alpha_rho)
#volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
 #                                                 volume_flux_dg=volume_flux,
  #                                                volume_flux_fv=surface_flux)
#solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 0.5, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000, periodicity=(false,false))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, #source_terms=source_terms_gravity,
                boundary_conditions=boundary_conditions)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.4)

time_series = TimeSeriesCallback(semi, [(0.4, 0.9)];
                                 interval=5,
                                 solution_variables=cons2prim)#,
                                 #filename="tseries.h5")

function save_my_plot_density(plot_data, variable_names;
                              show_mesh=false, plot_arguments=Dict{Symbol,Any}(),
                              time=nothing, timestep=nothing)
    
    alpha_rho_data = plot_data["alpha_rho"]
  
    title = @sprintf("alpha_rho | 4th order DG | t = %3.2f", time)
    
    Plots.plot(alpha_rho_data, 
               clim=(1000.0,1035.92), 
               title=title,titlefontsize=9, 
               dpi=300,
               )
  
    #Plots.plot!(getmesh(plot_data),linewidth=0.4)
  
    # Determine filename and save plot
    filename = joinpath("out", @sprintf("solution2_%06d.png", timestep))
    Plots.savefig(filename)
end

visualization_callback = VisualizationCallback(; interval=500,
                          solution_variables=cons2cons,
                          #variable_names=["rho"],
                          show_mesh=false,
                          #plot_data_creator=PlotData2D,
                          plot_creator=save_my_plot_density,
                          )

callbacks = CallbackSet(stepsize_callback, alive_callback, time_series)#,visualization_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), maxiters=1e7,
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

deg = 3
level = 3
sol_all_var = sol[2]
sol_all_var_0 = sol[1]

rho_vec = zeros((deg + 1)^2 * 2^level * 2^level)
rho0_vec = zeros((deg + 1)^2 * 2^level * 2^level)

v1_vec = zeros((deg + 1)^2 * 2^level * 2^level)
v1_0_vec = zeros((deg + 1)^2 * 2^level * 2^level)

v2_vec = zeros((deg + 1)^2 * 2^level * 2^level)
v2_0_vec = zeros((deg + 1)^2 * 2^level * 2^level)

a_vec = zeros((deg + 1)^2 * 2^level * 2^level)
a_0_vec = zeros((deg + 1)^2 * 2^level * 2^level)

for i = 1:((deg + 1)^2 * 2^level * 2^level)
    rho_vec[i] = sol_all_var[(i-1)*5 + 1]/sol_all_var[(i-1)*5 + 4]
    rho0_vec[i] = sol_all_var_0[(i-1)*5 + 1]/sol_all_var_0[(i-1)*5 + 4]
    
    v1_vec[i] = sol_all_var[(i-1)*5 + 2]/sol_all_var[(i-1)*5 + 1]
    v1_0_vec[i] = sol_all_var_0[(i-1)*5 + 2]/sol_all_var_0[(i-1)*5 + 1]
    
    v2_vec[i] = sol_all_var[(i-1)*5 + 3]/sol_all_var[(i-1)*5 + 1]
    v2_0_vec[i] = sol_all_var_0[(i-1)*5 + 3]/sol_all_var_0[(i-1)*5 + 1]
    
    a_vec[i] = sol_all_var[(i-1)*5 + 4]
    a_0_vec[i] = sol_all_var_0[(i-1)*5 + 4]
end


println("max. Fehler über erste Variable:  ", norm((rho0_vec - rho_vec),Inf))
println("max. Fehler über zweite Variable:  ", norm((v1_0_vec - v1_vec),Inf))
println("max. über dritte Variable:  ", norm((v2_0_vec - v2_vec),Inf))
println("max. über vierte Variable:  ", norm((a_0_vec - a_vec),Inf))
