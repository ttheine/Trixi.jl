
using Downloads: download
using OrdinaryDiffEq
using Revise
using Trixi
using Plots
using Printf
using LinearAlgebra

###############################################################################
#

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

boundary_condition = Dict(
  :all => boundary_condition_wall
)



volume_flux = (flux_central, flux_nonconservative_gaburro)
surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro)

solver = DGSEM(polydeg=3, surface_flux=surface_flux,
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-5.0, 5 * s - 5.0)
f2(s) = SVector( 5.0, 5 * s + 5.0)
f3(s) = SVector(5 * s, -5.0 + 5 * sin(0.5 * pi * s))
f4(s) = SVector(5 * s,  5.0 + 5 * sin(0.5 * pi * s))
faces = (f1, f2, f3, f4)

# This creates a mapping that transforms [-1, 1]^2 to the domain with the faces defined above.
# It generally doesn't work for meshes loaded from mesh files because these can be meshes
# of arbitrary domains, but the mesh below is specifically built on the domain [-1, 1]^2.
Trixi.validate_faces(faces)
mapping_flag = Trixi.transfinite_mapping(faces)

# Unstructured mesh with 24 cells of the square domain [-1, 1]^n
mesh_file = joinpath(@__DIR__, "square_unstructured_2.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
                              mesh_file)

mesh = P4estMesh{2}(mesh_file, polydeg=3,
                    mapping=mapping_flag,
                    initial_refinement_level=1)

                                    
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms = source_terms_gravity,
                                    boundary_conditions=boundary_condition)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                      base_level=1,
                                      med_level=2, med_threshold=0.1,
                                      max_level=3, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.7)

function save_my_plot_density(plot_data, variable_names;
                              show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
                              time=nothing, timestep=nothing)
    
    alpha_rho_data = plot_data["alpha_rho"]
  
    title = @sprintf("alpha_rho | 4th order DG | t = %3.2f", time)
    
    Plots.plot(alpha_rho_data, 
               #clim=(1000.0,1035.0), 
               title=title,titlefontsize=9, 
               dpi=300,
               )
  
    #Plots.plot!(getmesh(plot_data),linewidth=0.4)
  
    # Determine filename and save plot
    filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
    Plots.savefig(filename)
end

visualization_callback = VisualizationCallback(; interval=1000,
                            solution_variables=cons2cons,
                            #variable_names=["rho"],
                            show_mesh=false,
                            #plot_data_creator=PlotData2D,
                            plot_creator=save_my_plot_density,
                            )

callbacks = CallbackSet(summary_callback, alive_callback,
                        amr_callback, stepsize_callback, visualization_callback);


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary


# Gitter, 7680 Elemente in Sol[1] -> /5 = 1536 Punkte an denen ausgewertet wird 
sol_all_var = sol[2]
sol_all_var_0 = sol[1]

rho_vec = zeros(1536)
rho0_vec = zeros(1536)

v1_vec = zeros(1536)
v1_0_vec = zeros(1536)

v2_vec = zeros(1536)
v2_0_vec = zeros(1536)

a_vec = zeros(1536)
a_0_vec = zeros(1536)

for i = 1:(1536)
    rho_vec[i] = sol_all_var[(i-1)*5 + 1]
    rho0_vec[i] = sol_all_var_0[(i-1)*5 + 1]
    
    v1_vec[i] = sol_all_var[(i-1)*5 + 2]
    v1_0_vec[i] = sol_all_var_0[(i-1)*5 + 2]
    
    v2_vec[i] = sol_all_var[(i-1)*5 + 3]
    v2_0_vec[i] = sol_all_var_0[(i-1)*5 + 3]
    
    a_vec[i] = sol_all_var[(i-1)*5 + 4]
    a_0_vec[i] = sol_all_var_0[(i-1)*5 + 4]
end

println("Fehler über erste Variable:  ", norm((rho0_vec - rho_vec),Inf))
println("Fehler über zweite Variable:  ", norm((v1_0_vec - v1_vec),Inf))
println("Fehler über dritte Variable:  ", norm((v2_0_vec - v2_vec),Inf))
println("Fehler über vierte Variable:  ", norm((a_0_vec - a_vec),Inf))
