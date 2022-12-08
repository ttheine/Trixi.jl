
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations2D(1.00001)


function initial_condition_c(x, t, equations::CompressibleEulerEquations2D)
  if x[2] < 0.5
    rho = 50.5
  else
    rho = 50.0
  end
  rho = 100 * exp(- x[2])

  rho_v1 = 0.0
  rho_v2 = 0.0
  p = 1000.0 * rho
  rho_e = p / (equations.gamma - 1)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end

function initial_condition_rest(x, t, equations::CompressibleEulerEquations2D)
  rho = 1000.0 * exp(-(9.81 * 1000.0/(2.78*10^5)) *(x[2] - 1.0))
  rho_v1 = 0.0
  rho_v2 = 0.0
  p = 2.78*10^5 * (exp(-(9.81 * 1000.0/(2.78*10^5)) *(x[2] - 1.0)) - 1.0)
  rho_e = p / (equations.gamma - 1)
  
  return SVector(rho, rho_v1, rho_v2, rho_e)
end

function source_terms_gravityy(u, x, t, equations::CompressibleEulerEquations2D)
  du1 = 0.0
  du2 = 0.0
  du3 = -u[1] * 9.81
  du4 = 0.0

  return SVector(du1, du2, du3, du4)
end

initial_condition = initial_condition_c

boundary_conditions = (x_neg=boundary_condition_slip_wall,
                       x_pos=boundary_condition_slip_wall,
                       y_neg=boundary_condition_slip_wall,
                       y_pos=boundary_condition_slip_wall,)

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-0.5, 0.0)
coordinates_max = ( 0.5,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=10_000,
                periodicity=true)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,source_terms=source_terms_gravityy, boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)


stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
