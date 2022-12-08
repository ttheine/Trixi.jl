using OrdinaryDiffEq
using Trixi
using LinearAlgebra


###############################################################################
# simulation of a new year's firecracker between two houses


equations = AcousticPerturbationEquations2D(v_mean_global=(0.0, 0.0), c_mean_global=1.0, rho_mean_global=1.0)

function initial_condition_silvester(x, t, equations::AcousticPerturbationEquations2D)
    v1 = 0.0

    if (0.4 <= x[1] <= 0.6) && (0.1 <= x[2] <= 0.4) 
        v2 = 1.0
    else
        v2 = 0.0
    end

    if (0.4 <= x[1] <= 0.6) & (0.1 <= x[2] <= 0.4)
        p = 3 * exp(-((x[1] - 0.5)^2 + (x[2] - 0.25)^2) / (2 * 10^(-2))) + 2
    else
        p = 2.0
    end

    return SVector(v1, v2, p, global_mean_vars(equations)...)
end

initial_condition = initial_condition_silvester

function boundary_condition_house(u_inner, orientation, direction, x, t, surface_flux_function, equations::AcousticPerturbationEquations2D)
    # Boundary state is equal to the inner state except for the perturbed velocity. For boundaries
    # in the -x/+x direction, we multiply the perturbed velocity in the x direction by -1.
    # Similarly, for boundaries in the -y/+y direction, we multiply the perturbed velocity in the
    # y direction by -1
    if direction in (1, 2) # x direction
        u_boundary = SVector(-u_inner[1], u_inner[2], u_inner[3], cons2mean(u_inner, equations)...)
    else # y direction
        u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], cons2mean(u_inner, equations)...)
    end
    
    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

function boundary_condition_house(u_inner, normal_direction::AbstractVector, x, t, surface_flux_function, equations::AcousticPerturbationEquations2D)
    # normalize the outward pointing direction
    normal = normal_direction / norm(normal_direction)

    # compute the normal perturbed velocity
    u_normal = normal[1] * u_inner[1] + normal[2] * u_inner[2]

    # create the "external" boundary solution state
    u_boundary = SVector(u_inner[1] - 2.0 * u_normal * normal[1],
                       u_inner[2] - 2.0 * u_normal * normal[2],
                       u_inner[3], cons2mean(u_inner, equations)...)

    # calculate the boundary flux
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux
end


function outer_boundaries(x, t, equations::AcousticPerturbationEquations2D)
    return SVector(0.0, 0.0, 2.0, global_mean_vars(equations)...)
end

boundary_condition_outer_boundaries = BoundaryConditionDirichlet(outer_boundaries)

boundary_conditions = Dict( :Bottom  => boundary_condition_house,
                            :HouseWallLeft  => boundary_condition_house,
                            :HouseWallRight  => boundary_condition_house,
                            :HouseTopLeft  => boundary_condition_house,
                            :HouseTopRight  => boundary_condition_house,
                            :Top     => boundary_condition_outer_boundaries,
                            :Right   => boundary_condition_outer_boundaries,
                            :Left    => boundary_condition_outer_boundaries)


# Create DG solver with polynomial degree and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=6, surface_flux=flux_lax_friedrichs)

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# create the unstructured mesh from your mesh file
#mesh_file = joinpath("out", "silvester.mesh")
mesh_file = joinpath("out", "silvester_2.mesh")

mesh = UnstructuredMesh2D(mesh_file)
#mesh = P4estMesh{2}(mesh_file)

# Create semidiscretization with all spatial discretization-related components
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)



###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.5
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.4)                                     
# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)


##############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-8, reltol=1.0e-8, dt=1.0, save_everystep=false, callback=callbacks);

summary_callback()