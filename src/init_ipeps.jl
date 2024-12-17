"""
    init_ipeps(;atype = Array, Ni::Int, Nj::Int, D::Int)

return a random `ipeps` with bond dimension `D` and physical dimension 2.
"""
function init_ipeps(;atype = Array, file=nothing, D::Int, d::Int, Ni::Int, Nj::Int)
    if file !== nothing
        A = load(file, "bcipeps")
    else
        A = rand(ComplexF64, D,D,D,D,d,Ni,Nj)
        A /= norm(A)
    end
    return atype(A)
end

function init_ipeps_form_small_spin(;atype = Array, file=nothing, D::Int, d::Int, Ni::Int, Nj::Int)
    if file !== nothing
        Ao = load(file, "bcipeps")
        A = zeros(ComplexF64, D,D,D,D,d,Ni,Nj)
        A[:,:,:,:,[6,7,10,11],:,:] = Ao
    else
        A = rand(ComplexF64, D,D,D,D,d,Ni,Nj)
        A /= norm(A)
    end
    return atype(A)
end