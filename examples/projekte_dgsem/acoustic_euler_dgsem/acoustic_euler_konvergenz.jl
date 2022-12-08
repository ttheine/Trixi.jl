using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the acoustic perturbation equations

function calc_error(deg, level)
    equations = AcousticEulerEquations2D(rho=1.0, lambda=1.0)

    function initial_condition_test(x, t, equations::AcousticEulerEquations2D)
        v1 = 2 + sin(2* pi * x[1])
        v2 = 4 + cos(2* pi * x[2])
        p = 6 + sin(2 * pi * x[1]) + cos(2 * pi * x[2])

        return SVector(v1, v2, p)
    end

    initial_condition = initial_condition_test


    # Create DG solver with polynomial degree and (local) Lax-Friedrichs/Rusanov flux as surface flux
    solver = DGSEM(polydeg=deg, surface_flux=flux_lax_friedrichs)

    coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
    coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

    # Create a uniformly refined mesh with periodic boundaries
    mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=level,
                n_cells_max=30_000) # set maximum capacity of tree data structure


    # A semidiscretization collects data structures and functions for the spatial discretization
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


    ###############################################################################
    # ODE solvers, callbacks etc.

    # Create ODE problem with time span from 0.0 to 1.0
    tspan = (0.0, 1.0)
    ode = semidiscretize(semi, tspan)

    stepsize_callback = StepsizeCallback(cfl=0.1)                                     

    # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver

    callbacks = CallbackSet(stepsize_callback)

    ###############################################################################
    # run the simulation

    sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-10, reltol=1.0e-10, dt=1.0, save_everystep=false, callback=callbacks);
    

    error_all = abs.(sol[1] - sol[2])

    error_p = zeros(((deg + 1) * 2^level)^2)

    for i = 1:((deg + 1) * 2^level)^2
        error_p[i] = error_all[(i-1)*3 + 3]
    end

    return norm(error_p, Inf)
end

function konvergenz_tabelle(deg)
    
    polynomdeg = deg
    N_Q = [1,2,3,4,5]
    if deg == 7
        N_Q = [1,2,3,4]
    end
    error = zeros(length(N_Q))
    eoc = zeros(length(N_Q))

    global j = 1
    for level in N_Q
    
        error[j] = calc_error(polynomdeg, level)
    
        if j > 1
            eoc[j] = log(error[j]/error[j-1])/log(2^(j-2)/2^(j-1))
        end
        global j += 1
    end

    println("Polynomgrad =  ", polynomdeg)
    println("------------------------------------------------------------------")
    println("|   N_Q   |            Error           |           EOC           |")
    println("------------------------------------------------------------------")

    for i = 1:length(N_Q)
        println(2^N_Q[i],"  ", error[i],"  ", eoc[i])
    end

end