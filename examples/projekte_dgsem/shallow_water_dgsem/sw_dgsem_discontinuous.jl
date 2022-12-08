
using OrdinaryDiffEq
using Trixi

###############################################################################
# Entropie-Analyse

equations = ShallowWaterEquations1D(gravity_constant=9.812)

function initial_condition_entropy(x, t, equations)

    # bottom topography
    if abs(x[1] - 10) <= 2
        b = sin(pi/4 * x[1])
    else
        b = 0.0
    end

    if x[1] <= 10
        h = 3.0 - b
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
solver = DGSEM(polydeg=7, surface_flux=(flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################

# Get the TreeMesh and setup a periodic mesh

coordinates_min = 0.0
coordinates_max = 20.0
level = 2
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=level,
                n_cells_max=10_000)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_entropy, solver)

###############################################################################
# ODE solver

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)





###############################################################################
# Workaround to set a discontinuous bottom topography and initial condition for debugging and testing.

# alternative version of the initial conditinon used to setup a truly discontinuous
# bottom topography function and initial condition for this academic testcase of entropy conservation.
# The errors from the analysis callback are not important but `∑∂S/∂U ⋅ Uₜ` should be around machine roundoff
# In contrast to the usual signature of initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as intended
# only for the TreeMesh1D with `initial_refinement_level=4`.
function initial_condition_ec_discontinuous_bottom(x, t, element_id, equations::ShallowWaterEquations1D)
    # bottom topography
    if abs(x[1] - 10) <= 2
        b = sin(pi/4 * x[1])
    else
        b = 0.0
    end

    if x[1] <= 10
        h = 3.0 - b
    else
        h = 2.5 - b
    end

    # setup the discontinuous water height and velocity
    if element_id == 2^(level-1) + 1
      h = 2.5 - b
    end

    v = 0.0
    H = h + b
  
    return prim2cons(SVector(H, v, b), equations)
end
  
# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
    for i in eachnode(semi.solver)
        x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations, semi.solver, i, element)
        u_node = initial_condition_ec_discontinuous_bottom(x_node, first(tspan), element, equations)
        Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, element)
    end
end


# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy, energy_total))

alive_callback = AliveCallback(analysis_interval=analysis_interval)


stepsize_callback = StepsizeCallback(cfl=0.01)

visualization = VisualizationCallback(interval=5000, plot_data_creator=PlotData1D)


callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback, visualization)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-10, reltol=1.0e-10,
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

println("Fehler in der Gesamtenergie:  ", abs(Trixi.integrate(energy_total, sol.u[end], semi) - Trixi.integrate(energy_total, sol.u[1], semi)))
