# # MantisWrappers
#
# This module wraps the Mantis FEM library to provide a convenient interface for
# particle-to-B-spline projection, entropy computation, and Landau collision operators
# on a 2D Cartesian velocity domain.

module MantisWrappers

using Mantis
using LinearAlgebra

include("parameters.jl")

# ## Geometry and function spaces
#
# We build a 2D tensor-product B-spline space on the velocity domain [V_MIN, V_MAX]².
# The 0-form space X⁰ represents scalar fields f(v₁, v₂) = Σᵢ fᵢ φᵢ(v).

const bp = LinRange(V_MIN, V_MAX, N_ELEM + 1)

const geo_1d = Geometry.CartesianGeometry((bp,))
const B_1d = FunctionSpaces.BSplineSpace(geo_1d, P_DEG, K_REG)
const TP = FunctionSpaces.TensorProductSpace((B_1d, B_1d), Geometry.CartesianGeometry)
const X⁰ = Forms.FormSpace(0, TP, "f")

const n_dofs = Forms.get_num_basis(X⁰) # number of basis functions (degrees of freedom)
const geo_2d = Forms.get_geometry(X⁰)
const n_elements = Geometry.get_num_elements(geo_2d)
const lin_indices = LinearIndices((N_ELEM, N_ELEM))

# ## Mass matrix assembly
#
# The mass matrix M_{ij} = ∫ φᵢ φⱼ dv is assembled using the Mantis weak-form pipeline.
# We use a zero forcing term (AnalyticalFormField) as a dummy — only the bilinear form
# ∫ v⁰ ∧ ★(u⁰) is needed. The LU factorization is stored for repeated solves.

const M_lu = let
    qrule = Quadrature.tensor_product_rule((P_DEG + 1, P_DEG + 1), Quadrature.gauss_legendre)
    dΩ = Quadrature.StandardQuadrature(qrule, n_elements)
    f_zero = Forms.AnalyticalFormField(0, x -> [zeros(size(x, 1))], geo_2d, "0")
    wfi = Assemblers.WeakFormInputs(X⁰, f_zero)
    v⁰ = Assemblers.get_test_form(wfi)
    u⁰ = Assemblers.get_trial_form(wfi)
    M_expr = ∫(v⁰ ∧ ★(u⁰), dΩ)
    M_wf = Assemblers.WeakForm(((M_expr,),), ((0,),), wfi)
    M, _ = Assemblers.assemble(M_wf)
    lu(M)
end

# ## Quadrature rule for integration
#
# A separate quadrature rule used for numerical integration of entropy and the r-vector.

const _qrule_integrate = Quadrature.tensor_product_rule((N_QUAD, N_QUAD), Quadrature.gauss_legendre)

# ## Particle location
#
# Given a particle at (v₁, v₂), find which element it belongs to and compute the
# reference (canonical) coordinates ξ ∈ [0,1]² within that element.

struct ParticleLocation
    elem_id::Int
    xi::Points.CartesianPoints
    h1::Float64   # element width in v₁ direction
    h2::Float64   # element width in v₂ direction
end

function locate_particle(v1, v2)
    (v1 <= V_MIN || v1 >= V_MAX || v2 <= V_MIN || v2 >= V_MAX) && return nothing # TODO: type instable
    i = searchsortedlast(bp, v1)
    j = searchsortedlast(bp, v2)
    h1 = bp[i+1] - bp[i]
    h2 = bp[j+1] - bp[j]
    ξ1 = (v1 - bp[i]) / h1
    ξ2 = (v2 - bp[j]) / h2
    return ParticleLocation(lin_indices[i, j], Points.CartesianPoints(([ξ1], [ξ2])), h1, h2)
end

# ## Evaluate wrappers
#
# Thin wrappers around Mantis evaluation routines:
#   - evaluate(FormSpace, elem_id)       → basis values at quadrature nodes
#   - evaluate(FormField, elem_id)       → field values at quadrature nodes
#   - evaluate(FormSpace, ParticleLocation) → basis values at a single particle position

evaluate(fs::Forms.AbstractFormSpace, elem_id::Int) = Forms.evaluate(fs, elem_id, _qrule_integrate.nodes)
evaluate(ff::Forms.AbstractFormField, e::Int) = Forms.evaluate(ff, e, _qrule_integrate.nodes)
evaluate(fs::Forms.AbstractFormSpace, loc::ParticleLocation) = Forms.evaluate(fs, loc.elem_id, loc.xi)

# Evaluate basis function derivatives in canonical coordinates.
# Returns local_basis[deriv_order+1][deriv_index][component][point, basis].
# For a 2D 0-form: local_basis[2][1][1] = ∂φ/∂ξ₁, local_basis[2][2][1] = ∂φ/∂ξ₂.

evaluate_basis_derivatives(elem_id::Int, xi, nderivatives) =
    Forms._evaluate_form_in_canonical_coordinates(X⁰, elem_id, xi, nderivatives)
evaluate_basis_derivatives(loc::ParticleLocation, nderivatives) =
    Forms._evaluate_form_in_canonical_coordinates(X⁰, loc.elem_id, loc.xi, nderivatives)

# Build a FormField from coefficient vector: f_s(v) = Σᵢ fᵢ φᵢ(v).

build_field(coeffs) = Forms.build_form_field(X⁰, coeffs)

# Element Jacobian determinant (uniform Cartesian grid → same for all elements).

element_measure(e) = Geometry.get_element_measure(geo_2d, e)
const jac = element_measure(1)

export n_dofs, M_lu
export evaluate, build_field

# ## Physics routines
#
# The actual computation functions (L² projection, entropy, entropy gradient, collision
# operator) are defined in functions.jl.

include("functions.jl")

export compute_entropy, compute_r!, compute_G!, compute_collision!, l2_project!

end # module
