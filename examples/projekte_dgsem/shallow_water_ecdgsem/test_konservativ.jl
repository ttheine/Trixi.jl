
using OrdinaryDiffEq
using Trixi

###############################################################################
# Konservativitätsanalyse

equations = ShallowWaterEquations1D(gravity_constant=9.812)
deg = 3
level = 10   # -> keine Instabilitätsmeldung ohne callbacks

function initial_condition_conservative(x, t, equations)

    h = sin(pi/5 * x[1]) + 4.0
    v = 1.0
    # bottom topography
    if abs(x[1] - 10) <= 2
        b = sin(pi/4 * x[1])
    else
        b = 0.0
    end
    H = h + b

    return prim2cons(SVector(H, v, b), equations)
end


###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg=deg, surface_flux=surface_flux, volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = 0.0
coordinates_max = 20.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=level,
                n_cells_max=10_000)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_conservative, solver)

###############################################################################
# ODE solver

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

stepsize_callback = StepsizeCallback(cfl=0.5)

visualization = VisualizationCallback(interval=1000, plot_data_creator=PlotData1D)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback, visualization)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, abstol=1.0e-8, reltol=1.0e-8, save_everystep=false, callback=callbacks);


sol_all_var = sol[2]
sol_all_var_0 = sol[1]

h_vec = zeros((deg + 1) * 2^level)
h0_vec = zeros((deg + 1) * 2^level)

h_v_vec = zeros((deg + 1) * 2^level)
h_v0_vec = zeros((deg + 1) * 2^level)

for i = 1:((deg + 1) * 2^level)
    h_vec[i] = sol_all_var[(i-1)*3 + 1]
    h0_vec[i] = sol_all_var_0[(i-1)*3 + 1]
    h_v_vec[i] = sol_all_var[(i-1)*3 + 2]
    h_v0_vec[i] = sol_all_var_0[(i-1)*3 + 2]
end

x_LGL, w_LGL = Trixi.gauss_lobatto_nodes_weights(deg+1)

integral_h = 0 
integral_h0 = 0
integral_h_v = 0
integral_h_v0 = 0
for i in 1:length(h_vec)
    for j in 1:(deg + 1)
        global integral_h += h_vec[i] * w_LGL[j] * (coordinates_max-coordinates_min)/(2^level)*0.5
        global integral_h0 += h0_vec[i] * w_LGL[j] * (coordinates_max-coordinates_min)/(2^level)*0.5
        global integral_h_v += h_v_vec[i] * w_LGL[j] * (coordinates_max-coordinates_min)/(2^level)*0.5
        global integral_h_v0 += h_v0_vec[i] * w_LGL[j] * (coordinates_max-coordinates_min)/(2^level)*0.5
    end
end

println("Fehler Integral über erste Variable:  ", abs(integral_h - integral_h0))
println("Fehler Integral über zweite Variable:  ", abs(integral_h_v - integral_h_v0))