
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.812, H0=1.1)

function initial_condition_entropy(x, t, equations)
    # bottom topography
    if abs(x[1] - 10) <= 2
        b = sin(pi/4 * x[1])
    else
        b = 0.0
    end

    # height
    if x[1] <= 10
        h = 3.0 -b
    else
        h = 2.5 - b
    end

    v = 0.0
    H = h + b
    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_entropy

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_lax_friedrichs, hydrostatic_reconstruction_audusse_etal),
                flux_nonconservative_audusse_etal)
basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Create the TreeMesh 

coordinates_min = 0.0
coordinates_max = 20.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,
                n_cells_max=10_000,
                periodicity=false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                    extra_analysis_integrals=(energy_kinetic,
                                                              energy_internal,
                                                              lake_at_rest_error))

# Enable in-situ visualization with a new plot generated every 50 time steps
# and we explicitly pass that the plot data will be one-dimensional
visualization = VisualizationCallback(interval=400, plot_data_creator=PlotData1D)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, visualization, stepsize_callback)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, abstol=1.0e-10, reltol=1.0e-10,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

println("Fehler in der Gesamtenergie:  ", abs(Trixi.integrate(energy_total, sol.u[end], semi) - Trixi.integrate(energy_total, sol.u[1], semi)))