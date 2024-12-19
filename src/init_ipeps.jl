"""
    init_ipeps(;atype = Array, Ni::Int, Nj::Int, D::Int)

return a random `ipeps` with bond dimension `D` and physical dimension 2.
"""
function init_ipeps(;atype = Array, model, params::iPEPSOptimize{F}, No::Int=0, ifWp::Bool=false, ϵ::Real=1e-3, D::Int, χ::Int, Ni::Int, Nj::Int) where F
    d = Int(2*model.S + 1) 
    if No != 0
        file = joinpath(params.folder, "D$(D)_χ$(χ)/ipeps/ipeps_No.$(No).jld2")
        @info "load ipeps from file: $file"
        A = load(file, "bcipeps")
    else
        @info "generate random ipeps"
        if F == :merge
            A = rand(ComplexF64, D,D,D,D,d^2, Ni,Nj) .- 0.5
        elseif F == :brickwall
            A = rand(ComplexF64, D,1,D,D,d, Ni,Nj) .- 0.5
        end
        A /= norm(A)
    end
    if ifWp
        Wp = _arraytype(A)(bulid_Wp(model.S, params))
        A = bulid_A(A + randn(ComplexF64, size(A)) * ϵ, Wp, params) 
    end
    return atype(A)
end

function restriction_ipeps(A)
    # Ar = Zygote.Buffer(A)
    # Ni, Nj = size(A)[[6,7]]
    # # for j in 1:Nj, i in 1:Ni
    # #     # if i < 4
    # #     #     Ar[:,:,:,:,:,i,j] = A[:,:,:,:,:,i,j]
    # #     # else
    # #     #     Ar[:,:,:,:,:,i,j] = A[:,:,:,:,:,i-3,mod1(j+1, 2)]
    # #     # end
    # #     Ar[:,:,:,:,:,i,j] = A[:,:,:,:,:,1,1]
    # # end
    # Ar[:,:,:,:,:,1,1] = A[:,:,:,:,:,1,1]
    # Ar[:,:,:,:,:,1,2] = A[:,:,:,:,:,1,2]
    # Ar = copy(Ar)
    # return Ar/norm(Ar)
    return A/norm(A)
end

function init_ipeps_form_small_spin(;atype = Array, file, D::Int, d::Int, Ni::Int, Nj::Int)
    @info "load ipeps from file: $file"
    Ao = load(file, "bcipeps")
    A = zeros(ComplexF64, D,D,D,D,d,Ni,Nj)
    A[:,:,:,:,[6,7,10,11],:,:] = Ao
    return atype(A)
end