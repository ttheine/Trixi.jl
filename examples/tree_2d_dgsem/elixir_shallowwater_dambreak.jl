using OrdinaryDiffEq
using Trixi
using Plots
using Printf

equations = ShallowWaterEquations1D(gravity_constant=9.81)

function initial_condition_dambreak(x, t, equations)

    if(x[1] <= 0.0)
      # liquid domain
      #h = 0.375
      h = 1.5
      if(-0.5236 <= x[1] <= 0.5236)
        h = 1.125 + 0.375 * sin(3*x[1] + pi)
      end
    else
      h = 0.75
      if(-0.5236 <= x[1] <= 0.5236)
        h = 1.125 + 0.375 * sin(3*x[1] + pi)
      end
      #h = 0.1875
    end
    
    b = 0.0

    v = 0.0
    H = h + b

    return prim2cons(SVector(H, v, b), equations)
end

function initial_condition_step(x, t, equations)

    if(x[1] <= 0.0)
      # liquid domain
      #h = 0.375
      h = 1.4618
      b = 0.0
    else
      h = 0.30873
      b = 0.2
    end
    

    v = 0.0
    H = h + b

    return prim2cons(SVector(H, v, b), equations)
end

function initial_condition_dry_bed(x, t, equations)

    if(x[1] <= 0.0)
      # liquid domain
      #h = 0.375
      h = 1.4618
    else
      h = 0.001
    end
    
    b = 0.0

    v = 0.0
    H = h + b

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_step

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

#volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
#solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
 #              volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_lax_friedrichs, hydrostatic_reconstruction_audusse_etal),
                flux_nonconservative_audusse_etal)
basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################

# Get the TreeMesh and setup a periodic mesh

coordinates_min = -5
coordinates_max = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,
                n_cells_max=100_000, periodicity=false)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions = boundary_condition)

###############################################################################
# ODE solver

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)


# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 100

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.4)

#visualization = VisualizationCallback(interval=5000, plot_data_creator=PlotData1D)

function save_my_plot_h(plot_data, variable_names;
                              show_mesh=false, plot_arguments=Dict{Symbol,Any}(),
                              time=nothing, timestep=nothing)
    
    alpha_rho_data = plot_data["H"]
  
    title = @sprintf("H | 4th order DG | t = %3.2f", time)
    
    Plots.plot(alpha_rho_data, 
               ylim=(0.0,4.0), 
               title=title,titlefontsize=9, 
               dpi=300,
               )
  
    Plots.plot!(plot_data["b"])
  
    # Determine filename and save plot
    filename = joinpath("out", @sprintf("solution_shallow_%06d.png", timestep))
    Plots.savefig(filename)
end

visualization_callback = VisualizationCallback(; interval=500,
                          solution_variables=cons2prim,
                          show_mesh=false,
                          plot_data_creator=PlotData1D,
                          plot_creator=save_my_plot_h,
                          )

callbacks = CallbackSet(summary_callback, visualization_callback, alive_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
