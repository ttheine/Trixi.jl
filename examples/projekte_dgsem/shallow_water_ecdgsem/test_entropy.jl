
using OrdinaryDiffEq
using Trixi

###############################################################################
# Konservativitätsanalyse

equations = ShallowWaterEquations1D(gravity_constant=9.812)

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


###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg=3, surface_flux = surface_flux, volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################

# Get the TreeMesh and setup a periodic mesh

coordinates_min = 0.0
coordinates_max = 20.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,  
                n_cells_max=10_000)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_entropy, solver)

###############################################################################
# ODE solver

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

stepsize_callback = StepsizeCallback(cfl=0.1)

visualization = VisualizationCallback(interval=200, plot_data_creator=PlotData1D)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback, visualization)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, abstol=1.0e-10, reltol=1.0e-10, save_everystep=false, callback=callbacks);


println("Fehler in der Gesamtenergie:  ", abs(Trixi.integrate(energy_total, sol.u[end], semi) - Trixi.integrate(energy_total, sol.u[1], semi)))

