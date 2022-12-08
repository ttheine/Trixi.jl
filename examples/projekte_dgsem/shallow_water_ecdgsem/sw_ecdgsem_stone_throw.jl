using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=1.0, H0=1.1)

function initial_condition_stone_throw(x, t, equations::ShallowWaterEquations1D)
    # Calculate primitive variables
    H = equations.H0
    # v = 0.0 # for well-balanced test
    v = x[1] <= 5.0 ? -0.525 : 0.525 # for stone throw

    if abs(x[1]-10) <= 2
        b = sin(pi/4 * x[1])
    else
        b = 0.0
    end

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_stone_throw

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)

solver = DGSEM(polydeg=3, surface_flux=surface_flux, volume_integral=VolumeIntegralFluxDifferencing(volume_flux))


###############################################################################
# Create the TreeMesh for the domain [0, 20]

coordinates_min = 0.0
coordinates_max = 20.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,
                n_cells_max=10_000,
                periodicity=false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                    extra_analysis_integrals=(energy_kinetic,
                                                              energy_internal,
                                                              lake_at_rest_error))

# Enable in-situ visualization with a new plot generated every 50 time steps
# and we explicitly pass that the plot data will be one-dimensional
visualization = VisualizationCallback(interval=50, plot_data_creator=PlotData1D)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, visualization, stepsize_callback)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, abstol=1.0e-10, reltol=1.0e-10,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary