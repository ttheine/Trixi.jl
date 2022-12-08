# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


    @doc raw"""
    Gaburro1D(gamma)
    
    The Gaburro equations
    
    
    """
    struct Gaburro1D{RealT<:Real} <: AbstractGaburroEquations{1, 3}
      gamma::RealT
      k0::RealT
      rho_0::RealT

      function Gaburro1D(gamma, k0, rho_0)
        return new{typeof(gamma)}(gamma, k0, rho_0)
      end
    end
    
    #have_nonconservative_terms(::Gaburro1D) = Val(true)
    varnames(::typeof(cons2cons), ::Gaburro1D) = ("alpha_rho", "alpha_rho_v", "alpha")
    varnames(::typeof(cons2prim), ::Gaburro1D) = ("rho", "v", "alpha")
    
    
    # Set initial conditions at physical location `x` for time `t`
    """
        initial_condition_constant(x, t, equations::Gaburro1D)
    
    A constant initial condition to test free-stream preservation.
    """
    function initial_condition_constant(x, t, equations::Gaburro1D)
      alpha = 1.0
      alpha_rho = 0.1
      alpha_rho_v = -0.2
      return SVector(alpha_rho, alpha_rho_v, alpha)
    end

    # Calculate 1D flux for a single point
    @inline function flux(u, orientation::Integer, equations::Gaburro1D)
      alpha_rho, alpha_rho_v, alpha = u
      v = alpha_rho_v / alpha_rho
      rho = alpha_rho / alpha
      p = equations.k0 * ((rho/equations.rho_0)^equations.gamma - 1.0)
      
      f1 = alpha_rho * v
      f2 = alpha_rho * v^2 + alpha * p
      f3 = 0
      
      return SVector(f1, f2, f3)
    end

    #@inline function flux_nonconservative_gaburro(u_ll, u_rr, orientation::Integer, equations::Gaburro1D)
        # Pull the necessary left and right state information
     #   v_ll = u_ll[2]/u_ll[1]
      #  alpha_rr = u_rr[3]

       # z = zero(eltype(u_ll))

        #f = SVector(z, z, v_ll * alpha_rr)

        #return f
    #end

    
    # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
    # maximum velocity magnitude plus the maximum speed of sound
    @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::Gaburro1D)
      rho_ll, v_ll, alpha_ll = cons2prim(u_ll, equations)
      p_ll = pressure(u_ll, equations)
      rho_rr, v_rr, alpha_rr = cons2prim(u_rr, equations)
      p_rr = pressure(u_rr, equations)
    
      # Calculate sound speeds
      c_ll = sqrt(equations.gamma * equations.k0 / equations.rho_0)
      c_rr = sqrt(equations.gamma * equations.k0 / equations.rho_0)

      λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
    end
    
    
    # Calculate minimum and maximum wave speeds for HLL-type fluxes
    @inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                         equations::Gaburro1D)
      rho_ll, v_ll, alpha_ll = cons2prim(u_ll, equations)
      p_ll = pressure(u_ll, equations)
      rho_rr, v_rr, alpha_rr = cons2prim(u_rr, equations)
      p_rr = pressure(u_rr, equations)

      λ_min = v_ll - sqrt(equations.gamma * equations.k0 / equations.rho_0)
      λ_max = v_rr + sqrt(equations.gamma * equations.k0 / equations.rho_0)
    
      return λ_min, λ_max
    end
 
    
    @inline function max_abs_speeds(u, equations::Gaburro1D)
      rho, v, alpha = cons2prim(u, equations)
      p = pressure(u, equations)
      c = sqrt(equations.gamma * equations.k0 / equations.rho_0)
    
      return (abs(v) + c,)
    end
    
    
    # Convert conservative variables to primitive
    @inline function cons2prim(u, equations::Gaburro1D)
      alpha_rho, alpha_rho_v, alpha = u

      rho = alpha_rho / alpha
      v = alpha_rho_v / alpha_rho
    
      return SVector(rho, v, alpha)
    end
    
    
    
    # Convert primitive to conservative variables
    @inline function prim2cons(prim, equations::Gaburro1D)
      rho, v, alpha, = prim
      alpha_rho = rho * alpha
      alpha_rho_v = alpha_rho * v 
      return SVector(alpha_rho, alpha_rho_v, alpha)
    end
    
    
    @inline function density(u, equations::Gaburro1D)
     rho = u[1]/u[3]
     return rho
    end
    
    
    @inline function pressure(u, equations::Gaburro1D)
      alpha_rho, alpha_rho_v, alpha = u
      rho = alpha_rho / alpha
      p = equations.k0 * ((rho/equations.rho_0)^equations.gamma - 1)
      return p
    end
    
    
    @inline function density_pressure(u, equations::Gaburro1D)
     alpha_rho, alpha_rho_v, alpha = u
     rho = alpha_rho / alpha
     rho_times_p = equations.k0 * ((rho/equations.rho_0)^equations.gamma - 1) * rho
     return rho_times_p
    end
    
    
    
    end # @muladd
    