using OrdinaryDiffEq
using Revise
using Trixi
using Plots
using Printf

equations = Gaburro2D(1.0, 2.25*10^9, 1000.0, 0.0)

function initial_condition_elliptic(x, t, equations::Gaburro2D)
  
    # liquid domain
    if((x[1]^2 + x[2]^2) <= 1)
        rho = 1000.0
        alpha = 1.0 - 10^(-3)
        v1 = -100.0 * x[1]
        v2 = 100.0 * x[2]
    else
        rho = 1000.0
        v1 = 0.0
        v2 = 0.0
        alpha = 10^(-3)
    end
    phi = 0.0
    
    return prim2cons(SVector(rho, v1, v2, alpha, phi), equations)
end

initial_condition = initial_condition_elliptic

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

#volume_integral=VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Get the unstructured quad mesh from a file 
# create the unstructured mesh from your mesh file
mesh_file = joinpath("out", "tank_elliptic.mesh")

mesh = UnstructuredMesh2D(mesh_file)#, periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                boundary_conditions=boundary_condition)

tspan = (0.0, 0.0076)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.8)


function save_my_plot_density(plot_data, variable_names;
                              show_mesh=false, plot_arguments=Dict{Symbol,Any}(),
                              time=nothing, timestep=nothing)
    
    alpha_rho_data = plot_data["alpha_rho"]
  
    title = @sprintf("alpha_rho | 4th order DG | t = %3.4f", time)
    
    Plots.plot(alpha_rho_data, 
               clim=(0.0,1000.0), 
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

sol = solve(ode, CarpenterKennedy2N54( williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
