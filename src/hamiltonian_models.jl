const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -1im; 1im 0]
const σz = ComplexF64[1 0; 0 -1]
const id2 = ComplexF64[1 0; 0 1]

function const_Sx(S::Rational)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sx = zeros(ComplexF64, dims, dims)
    for j in 1:dims, i in 1:dims
        if abs(i-j) == 1
            Sx[i,j] = 1/2 * sqrt(S*(S+1)-ms[i]*ms[j]) 
        end
    end
    return Sx
end

function const_Sy(S::Rational)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sy = zeros(ComplexF64, dims, dims)
    for j in 1:dims, i in 1:dims
        if i-j == 1
            Sy[i,j] = -1/2/1im * sqrt(S*(S+1)-ms[i]*ms[j]) 
        elseif j-i == 1
            Sy[i,j] =  1/2/1im * sqrt(S*(S+1)-ms[i]*ms[j]) 
        end
    end
    return Sy
end

function const_Sz(S::Rational)
    dims = Int(2*S + 1)
    ms = [S-i+1 for i in 1:dims]
    Sz = zeros(ComplexF64, dims, dims)
    for i in 1:dims
        Sz[i,i] = ms[i]
    end
    return Sz
end

abstract type HamiltonianModel end

@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end
struct diaglocal{T<:Vector} <: HamiltonianModel 
    Ni::Int
    Nj::Int
    diag::T
end

"""
    diaglocal(diag::Vector)

return the 2-site Hamiltonian with single-body terms given
by the diagonal `diag`.
"""
function hamiltonian(model::diaglocal)
    diag = model.diag
    n = length(diag)
    h = ein"i -> ii"(diag)
    id = Matrix(I,n,n)
    reshape(h,n,n,1,1) .* reshape(id,1,1,n,n) .+ reshape(h,1,1,n,n) .* reshape(id,n,n,1,1)
end

@doc raw"
    TFIsing(hx::Real)

return a struct representing the transverse field ising model with magnetisation `hx`.
"
struct TFIsing{T<:Real} <: HamiltonianModel
    Ni::Int
    Nj::Int
    hx::T
end

"""
    hamiltonian(model::TFIsing)

return the transverse field ising hamiltonian for the provided `model` as a
two-site operator.
"""
function hamiltonian(model::TFIsing)
    hx = model.hx
    -2 * ein"ij,kl -> ijkl"(σz,σz) -
        hx/2 * ein"ij,kl -> ijkl"(σx, id2) -
        hx/2 * ein"ij,kl -> ijkl"(id2, σx)
end

@doc raw"
    Heisenberg(Jz::T,Jx::T,Jy::T) where {T<:Real}

return a struct representing the heisenberg model with magnetisation fields
`Jz`, `Jx` and `Jy`..
"
struct Heisenberg{T<:Real} <: HamiltonianModel
    Ni::Int
    Nj::Int
    Jz::T
    Jx::T
    Jy::T
end
Heisenberg(Ni,Nj) = Heisenberg(Ni,Nj,1.0,1.0,1.0)

"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    h = model.Jz * ein"ij,kl -> ijkl"(σz, σz) -
        model.Jx * ein"ij,kl -> ijkl"(σx, σx) -
        model.Jy * ein"ij,kl -> ijkl"(σy, σy)
    # h = ein"ijcd,kc,ld -> ijkl"(h,σx,σx')
    h / 8
end

@doc raw"
    Kitaev(Jx::T,Jz::T,Jy::T) where {T<:Real}

return a struct representing the Kitaev model with magnetisation fields
`Jx`, `Jy` and `Jz`..
"
struct Kitaev <: HamiltonianModel
    S::Rational
    Jz::Real
    Jx::Real
    Jy::Real
end
Kitaev() = Kitaev(1//2, -1.0, -1.0, -1.0)

"""
    hamiltonian(model::Kitaev)

return the Kitaev hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Kitaev)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    hx = model.Jx * ein"ij,kl -> ijkl"(Sx, Sx)
    hy = model.Jy * ein"ij,kl -> ijkl"(Sy, Sy)
    hz = model.Jz * ein"ij,kl -> ijkl"(Sz, Sz)
    return hx, hy, hz
end

@doc raw"
    Kitaev_Heisenberg{T<:Real} <: HamiltonianModel

return a struct representing the Kitaev_Heisenberg model with interaction factor
`ϕ` degree
"
struct Kitaev_Heisenberg{T<:Real} <: HamiltonianModel
    ϕ::T
end

"""
    hamiltonian(model::Kitaev_Heisenberg)

return the Kitaev_Heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Kitaev_Heisenberg)
    op = ein"ij,kl -> ijkl"
    Heisenberg = cos(model.ϕ / 180 * pi) / 2 * (op(σz, σz) +
                                 op(σx, σx) +
                                 op(σy, σy) )
    hx = Heisenberg + sin(model.ϕ / 180 * pi) * op(σx, σx)
    hy = Heisenberg + sin(model.ϕ / 180 * pi) * op(σy, σy)
    hz = Heisenberg + sin(model.ϕ / 180 * pi) * op(σz, σz)
    hx / 8, hy / 8, hz / 8
end

@doc raw"
    K_J_Γ_Γ′{T<:Real} <: HamiltonianModel

return a struct representing the K_J_Γ_Γ′ model
"
struct K_J_Γ_Γ′{T<:Real} <: HamiltonianModel
    S::Real
    K::T
    J::T
    Γ::T
    Γ′::T
end

"""
    hamiltonian(model::K_J_Γ_Γ)

return the K_J_Γ_Γ′ hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::K_J_Γ_Γ′)
    op = ein"ij,kl -> ijkl"
    Sz = const_Sz(model.S)
    Sx = const_Sx(model.S)
    Sy = const_Sy(model.S)
    Heisenberg = model.J * (op(Sz, Sz) +
                            op(Sx, Sx) +
                            op(Sy, Sy) )
    hx = Heisenberg + model.K * op(Sx, Sx) + model.Γ * (op(Sy, Sz) + op(Sz, Sy)) + model.Γ′ * (op(Sx, Sy) + op(Sy, Sx) + op(Sz, Sx) + op(Sx, Sz))
    hy = Heisenberg + model.K * op(Sy, Sy) + model.Γ * (op(Sx, Sz) + op(Sz, Sx)) + model.Γ′ * (op(Sy, Sx) + op(Sx, Sy) + op(Sz, Sy) + op(Sy, Sz))
    hz = Heisenberg + model.K * op(Sz, Sz) + model.Γ * (op(Sx, Sy) + op(Sy, Sx)) + model.Γ′ * (op(Sz, Sx) + op(Sx, Sz) + op(Sy, Sz) + op(Sz, Sy))
    hx / 2, hy / 2, hz / 2
end

@doc raw"
    K_Γ{T<:Real} <: HamiltonianModel

return a struct representing the K_Γ model
"
struct K_Γ{T<:Real} <: HamiltonianModel
    ϕ::T
end

"""
    hamiltonian(model::K_Γ)

return the K_Γ hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::K_Γ)
    op = ein"ij,kl -> ijkl"
    hx = -cos(model.ϕ * pi) * op(σx, σx) + sin(model.ϕ * pi) * (op(σy, σz) + op(σz, σy))
    hy = -cos(model.ϕ * pi) * op(σy, σy) + sin(model.ϕ * pi) * (op(σx, σz) + op(σz, σx))
    hz = -cos(model.ϕ * pi) * op(σz, σz) + sin(model.ϕ * pi) * (op(σx, σy) + op(σy, σx))
    hx / 8, hy / 8, hz / 8
end