using OrdinaryDiffEq
using Revise
using Trixi
using Plots
using Printf

equations = ThreeEquationModel2D(1.0, 1.0, 1.0, 9.81)

function initial_condition_convergence_test(x, t, equations::ThreeEquationModel2D)
  alpha_rho = 2 + 0.1 * sin(pi*(x[1] + x[2] -t))
  alpha_rho_v1 = 2 + 0.1 * sin(pi*(x[1] + x[2] -t))
  alpha_rho_v2 = 2 + 0.1 * sin(pi*(x[1] + x[2] -t))
  alpha = 1.0
  phi = x[2]
  return SVector(alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi)
end

initial_condition = initial_condition_convergence_test

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


solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro),
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))
#solver = DGSEM(basis, surface_flux, volume_integral)


coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (2.0, 2.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=400_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, 
                                    source_terms=source_terms_convergence_test)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.0)


callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54( williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
