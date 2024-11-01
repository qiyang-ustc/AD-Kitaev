using AD_Kitaev
using CUDA
using Random
using Printf

Random.seed!(100)
model   = K_J_Γ_Γ′(1.5, 1.0, 0.0, 0.0, 0.0)
folder  = "./../../../../data/xyzhang/AD_Kitaev/"
# folder = "./example/data/"
# folder  = "../data/AD_Kitaev/"
atype   = CuArray
D, χ    = 5, 60
tol     = 1e-10
maxiter = 50
miniter = 1
Ni, Nj  = 1, 1

fdirect = [1.0, 1.0, 1.0]
type    = "_random"

for targχ in 60:10:60, field in 0.0:0.01:0.0
    file = joinpath(folder, "$(Ni)x$(Nj)", "$(model)", "D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).jld2")
    if ispath(file)
        @show field
        observable(model, fdirect, field, type, folder, atype, D, χ, targχ, tol, maxiter, miniter, Ni, Nj; ifload = false)
    end
end