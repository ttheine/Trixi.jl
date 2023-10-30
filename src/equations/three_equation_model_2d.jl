# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
  ThreeEquationModel2D(gamma)
    
  The three equation model
    
    
  """
struct ThreeEquationModel2D{RealT<:Real} <: AbstractThreeEquationModel{2, 5}
  gamma::RealT
  k0::RealT
  rho_0::RealT
  gravity::RealT

  function ThreeEquationModel2D(gamma, k0, rho_0, gravity)
    return new{typeof(gamma)}(gamma, k0, rho_0, gravity)
  end
end


have_nonconservative_terms(::ThreeEquationModel2D) = True()
varnames(::typeof(cons2cons), ::ThreeEquationModel2D) = ("alpha_rho", "alpha_rho_v1", "alpha_rho_v2", "alpha", "phi")
varnames(::typeof(cons2prim), ::ThreeEquationModel2D) = ("rho", "v1", "v2", "alpha", "phi")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::ThreeEquationModel2D)
    
A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::ThreeEquationModel2D)
  alpha_rho = 1000.0
  alpha_rho_v1 = 0.0
  alpha_rho_v2 = 0.0
  alpha = 1.0
  phi = x[2]
  return SVector(alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi)
end

function source_terms_gravity(u, x, t, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha = u
  du1 = 0.0
  du2 = 0.0
  du3 = -alpha_rho * equations.gravity
  du4 = 0.0
  du5 = 0.0

  return SVector(du1, du2, du3, du4, du5)
end

function source_terms_convergence_test(u, x, t, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u
  du1 = 0.1 * pi * cos(pi*(x[1] + x[2] -t))
  du2 = 2 * 0.1 * pi * cos(pi*(x[1] + x[2] -t))
  du3 = 2 * 0.1 * pi * cos(pi*(x[1] + x[2] -t))
  du4 = 0.0
  du5 = 0.0

  return SVector(du1, du2, du3, du4, du5)
end


function boundary_condition_wall(u_inner, orientation, 
                                 direction, x, t,
                                 surface_flux_function,
                                 equations::Gaburro2D)

  # Boundary state is equal to the inner state except for the velocity. For boundaries
  # in the -x/+x direction, we multiply the velocity in the x direction by -1.
  # Similarly, for boundaries in the -y/+y direction, we multiply the velocity in the
  # y direction by -1
  if direction in (1, 2) # x direction
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5])
  else # y direction
    u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4], u_inner[5])
  end

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end

@inline function boundary_condition_wall(u_inner, normal_direction::AbstractVector,
                                         x, t,
                                         surface_flux_function,
                                         equations::ThreeEquationModel2D)
  
  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # compute the normal velocity
  u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]

  # create the "external" boundary solution state
  u_boundary = SVector(u_inner[1],
                       u_inner[2] - 2.0 * u_normal * normal[1],
                       u_inner[3] - 2.0 * u_normal * normal[2],
                       u_inner[4],
                       u_inner[5])

  # calculate the boundary flux
  flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

  return flux
end


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u
  v1 = alpha_rho_v1 / alpha_rho
  v2 = alpha_rho_v2 / alpha_rho
  p = pressure(u, equations)
  if orientation == 1
    f1 = alpha_rho_v1
    f2 = alpha_rho_v1 * v1 + alpha * p
    f3 = alpha_rho_v1 * v2
    f4 = 0.0
    f5 = 0.0
  else
    f1 = alpha_rho_v2
    f2 = alpha_rho_v1 * v2
    f3 = alpha_rho_v2 * v2 + alpha * p
    f4 = 0.0
    f5 = 0.0
  end
  return SVector(f1, f2, f3, f4, f5)
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equations::ThreeEquationModel2D)
  rho, v1, v2, alpha, phi = cons2prim(u, equations)

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  rho_v_normal = rho * v_normal
  p = pressure(u, equations)

  f1 = alpha * rho_v_normal
  f2 = alpha * rho_v_normal * v1 + alpha * p * normal_direction[1]
  f3 = alpha * rho_v_normal * v2 + alpha * p * normal_direction[2]
  f4 = 0.0
  f5 = 0.0
  
  return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_nonconservative_gaburro(u_ll, u_rr, orientation::Integer, equations::ThreeEquationModel2D)
  
  v1_ll = u_ll[2]/u_ll[1]
  v2_ll = u_ll[3]/u_ll[1]
  alpha_rr = u_rr[4]
  
  z = zero(eltype(u_ll))
  
  if orientation == 1
    f = SVector(z, z, z, v1_ll * alpha_rr, z)
  else
    f = SVector(z, z, z, v2_ll * alpha_rr, z)
  end
      
  return f
end


@inline function flux_nonconservative_gaburro(u_ll, u_rr,
                                              normal_direction_ll::AbstractVector,
                                              normal_direction_average::AbstractVector,
                                              equations::ThreeEquationModel2D)

  v1_ll = u_ll[2]/u_ll[1]
  v2_ll = u_ll[3]/u_ll[1]
  alpha_rr = u_rr[4]

  v_dot_n_ll = v1_ll * normal_direction_ll[1] + v2_ll * normal_direction_ll[2]
  
  
  z = zero(eltype(u_ll))

  f = SVector(z, z, z, v_dot_n_ll * alpha_rr, z)

  return f

end

@inline function flux_nonconservative_gaburro_well(u_ll, u_rr, orientation::Integer, equations::ThreeEquationModel2D)
  
  v1_ll = u_ll[2]/u_ll[1]
  v2_ll = u_ll[3]/u_ll[1]
  alpha_rr = u_rr[4]
  phi_ll = u_ll[5]
  phi_rr = u_rr[5]
  
  gravity = u_ll[1] * equations.gravity * phi_rr
  well_balanced = u_ll[1]/equations.rho_0 * equations.k0 * exp(equations.rho_0 * equations.gravity * phi_ll/equations.k0) * exp(-equations.rho_0 * equations.gravity * phi_rr/equations.k0)
  
  z = zero(eltype(u_ll))
  
  if orientation == 1
    f = SVector(z, z, z, v1_ll * alpha_rr, z)
  else
    f = SVector(z, z, -well_balanced, v2_ll * alpha_rr, z)
  end
      
  return f
end


@inline function flux_nonconservative_gaburro_well(u_ll, u_rr,
                                              normal_direction_ll::AbstractVector,
                                              normal_direction_average::AbstractVector,
                                              equations::ThreeEquationModel2D)

  v1_ll = u_ll[2]/u_ll[1]
  v2_ll = u_ll[3]/u_ll[1]
  alpha_rr = u_rr[4] #* normal_direction_average[1] + u_rr[4] * normal_direction_average[2]
  phi_ll = u_ll[5]
  phi_rr = u_rr[5]

  v_dot_n_ll = v1_ll * normal_direction_ll[1] + v2_ll * normal_direction_ll[2]
  
  well_balanced = u_ll[1]/equations.rho_0 * equations.k0 * exp(equations.rho_0 * equations.gravity * phi_ll/equations.k0) * exp(-equations.rho_0 * equations.gravity * phi_rr/equations.k0) * normal_direction_average[2]
  gravity = u_ll[1] * equations.gravity * u_rr[5] * normal_direction_average[2]
  
  z = zero(eltype(u_ll))

  f = SVector(z, z, -well_balanced, v_dot_n_ll * alpha_rr, z)

  return f

end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::ThreeEquationModel2D)
  rho_ll, v1_ll, v2_ll, alpha_ll, phi_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, alpha_rr, phi_rr = cons2prim(u_rr, equations)
  
  # Get the velocity value in the appropriate direction
  if orientation == 1
    v_ll = v1_ll
    v_rr = v1_rr
  else # orientation == 2
    v_ll = v2_ll
    v_rr = v2_rr
  end
  # Calculate sound speeds
  c_ll = sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_ll/equations.rho_0)^(equations.gamma - 1))
  c_rr = sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_rr/equations.rho_0)^(equations.gamma - 1))

  λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::ThreeEquationModel2D)
  rho_ll, v1_ll, v2_ll, alpha_ll, phi_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, alpha_rr, phi_rr = cons2prim(u_rr, equations)
  
  # Calculate normal velocities and sound speed
  # left
  v_ll = (  v1_ll * normal_direction[1]
          + v2_ll * normal_direction[2] )
  c_ll = sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_ll/equations.rho_0)^(equations.gamma - 1))
  # right
  v_rr = (  v1_rr * normal_direction[1]
          + v2_rr * normal_direction[2] )
  c_rr = sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_rr/equations.rho_0)^(equations.gamma - 1))
    
  return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end



# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::ThreeEquationModel2D)
  rho_ll, v1_ll, v2_ll, alpha_ll, phi_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, alpha_rr, phi_rr = cons2prim(u_rr, equations)

  if orientation == 1 # x-direction
    λ_min = v1_ll - sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_ll/equations.rho_0)^(equations.gamma - 1))
    λ_max = v1_rr + sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_rr/equations.rho_0)^(equations.gamma - 1))
  else # y-direction
    λ_min = v2_ll - sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_ll/equations.rho_0)^(equations.gamma - 1))
    λ_max = v2_rr + sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_rr/equations.rho_0)^(equations.gamma - 1))
  end
    
  return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                      equations::ThreeEquationModel2D)
  rho_ll, v1_ll, v2_ll, alpha_ll, phi_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, alpha_rr, phi_rr = cons2prim(u_rr, equations)

  v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  norm_ = norm(normal_direction)
  # The v_normals are already scaled by the norm
  λ_min = v_normal_ll - sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_ll/equations.rho_0)^(equations.gamma - 1)) * norm_
  λ_max = v_normal_rr + sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho_rr/equations.rho_0)^(equations.gamma - 1)) * norm_

  return λ_min, λ_max
end


@inline function max_abs_speeds(u, equations::ThreeEquationModel2D)
  rho, v1, v2, alpha, phi = cons2prim(u, equations)
  c = sqrt(equations.gamma * (equations.k0 / equations.rho_0) * (rho/equations.rho_0)^(equations.gamma - 1))
    
  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u

  rho = alpha_rho/alpha
  v1 = alpha_rho_v1 / alpha_rho
  v2 = alpha_rho_v2 / alpha_rho
    
  return SVector(rho, v1, v2, alpha, phi)
end

# Convert conservative variables to primitive
@inline function cons2entropy(u, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u

  rho = alpha_rho/alpha
  v1 = alpha_rho_v1 / alpha_rho
  v2 = alpha_rho_v2 / alpha_rho
    
  return SVector(rho, v1, v2, alpha, phi)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ThreeEquationModel2D)
  rho, v1, v2, alpha, phi = prim
  alpha_rho = rho * alpha
  alpha_rho_v1 = alpha_rho * v1
  alpha_rho_v2  = alpha_rho * v2 
  return SVector(alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi)
end

@inline function density(u, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u
  return alpha_rho/alpha
end

@inline function alpha_rho(u, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u
  return alpha_rho
end

@inline function pressure(u, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u
  if alpha < 0.1
    p = 0.0
  else
    p = equations.k0 * ((alpha_rho/equations.rho_0)^(equations.gamma) - 1)
  end
  return p
end

@inline function density_pressure(u, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u
  rho = alpha_rho / alpha
  rho_times_p = pressure(u,equations) * rho
  return rho_times_p
end

# Calculate the error for the "water-at-rest" test case 
@inline function water_at_rest_error(u, equations::ThreeEquationModel2D)
  alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u
  rho0 = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) * (phi - 1.0))
  return abs(alpha_rho/alpha - rho0)
end

end # @muladd  
