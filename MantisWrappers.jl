module MantisWrappers

using Mantis
using LinearAlgebra

include("parameters.jl")

const bp = LinRange(V_MIN, V_MAX, N_ELEM + 1)

const geo_1d = Geometry.CartesianGeometry((bp,))
const B_1d = FunctionSpaces.BSplineSpace(geo_1d, P_DEG, K_REG)
const TP = FunctionSpaces.TensorProductSpace((B_1d, B_1d), Geometry.CartesianGeometry)
const X⁰ = Forms.FormSpace(0, TP, "f")

const n_dofs = Forms.get_num_basis(X⁰)
const geo_2d = Forms.get_geometry(X⁰)
const n_elements = Geometry.get_num_elements(geo_2d)
const lin_indices = LinearIndices((N_ELEM, N_ELEM))

const M_mat, M_lu = let
    qrule = Quadrature.tensor_product_rule((P_DEG + 1, P_DEG + 1), Quadrature.gauss_legendre)
    dΩ = Quadrature.StandardQuadrature(qrule, n_elements)
    f_zero = Forms.AnalyticalFormField(0, x -> [zeros(size(x, 1))], geo_2d, "0")
    wfi = Assemblers.WeakFormInputs(X⁰, f_zero)
    v⁰ = Assemblers.get_test_form(wfi)
    u⁰ = Assemblers.get_trial_form(wfi)
    M_expr = ∫(v⁰ ∧ ★(u⁰), dΩ)
    M_wf = Assemblers.WeakForm(((M_expr,),), ((0,),), wfi)
    M, _ = Assemblers.assemble(M_wf)
    M, lu(M)
end

const _qrule_integrate = Quadrature.tensor_product_rule((N_QUAD, N_QUAD), Quadrature.gauss_legendre)

function locate_particle(v1, v2)
    (v1 <= V_MIN || v1 >= V_MAX || v2 <= V_MIN || v2 >= V_MAX) && return nothing
    i = searchsortedlast(bp, v1)
    j = searchsortedlast(bp, v2)
    h1 = bp[i+1] - bp[i]
    h2 = bp[j+1] - bp[j]
    ξ1 = (v1 - bp[i]) / h1
    ξ2 = (v2 - bp[j]) / h2
    return (; elem_id=lin_indices[i, j], ξ1, ξ2, h1, h2)
end


evaluate(fs::Forms.AbstractFormSpace, elem_id, xi) = Forms.evaluate(fs, elem_id, xi)
evaluate(ff::Forms.AbstractFormField, e, xi) = Forms.evaluate(ff, e, xi)

evaluate_basis_derivatives(elem_id, xi, nderivatives) =
    Forms._evaluate_form_in_canonical_coordinates(X⁰, elem_id, xi, nderivatives)

build_field(coeffs) = Forms.build_form_field(X⁰, coeffs)

element_measure(e) = Geometry.get_element_measure(geo_2d, e)

quadrature_nodes_weights() = (_qrule_integrate.nodes, _qrule_integrate.weights)

make_point(ξ1, ξ2) = Points.CartesianPoints(([ξ1], [ξ2]))

export X⁰, M_mat, M_lu, n_dofs, n_elements, geo_2d, bp
export locate_particle, make_point
export evaluate, evaluate_basis_derivatives, build_field
export element_measure, quadrature_nodes_weights

end # module
