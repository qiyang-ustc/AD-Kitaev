println(read(@__FILE__, String))

using AD_Kitaev
using Random
using CUDA
using TeneT
using LinearAlgebra

#####################################    parameters      ###################################
Random.seed!(42)
atype = trycuda() ? CuArray : Array
su_seed=3
Ni, Nj = 2, 6
D, χ = 5, 60
No = 0
S = 1
model = Kitaev(S,1.0,1.0,1.0)
Dz = 0.0
# method = :merge
method = :brickwall
folder = "data/$method/D$(D)/$model/$(Ni)x$(Nj)/"
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=true, 
                     ifsimple_eig=true,
                     maxiter=30, 
                     miniter=1,
                     maxiter_ad=3,
                     miniter_ad=3,
                     ifcheckpoint=true,
                     verbosity=3,
                     show_every=1
)
params = iPEPSOptimize{method}(boundary_alg=boundary_alg,
                               reuse_env=true, 
                               verbosity=4, 
                               maxiter=100,
                               tol=1e-10,
                               folder=folder
)
# A = init_ipeps(;atype, model, params, No, ifWp=false, ϵ = 5*1e-2, D, χ, Ni, Nj)
# A = AD_Kitaev.init_ipeps_spin111(;atype, model, params, No, ifWp=true, ϵ = 0, χ, Ni, Nj)
A = AD_Kitaev.init_ipeps_h5(;atype, model, file="./data/kitsShf:sikh2nfr$(su_seed)D$(D)D$(D).h5", D, Ni, Nj)
# A = norm(imag(A)) < 1E-10 ? real(A) : A
############################################################################################

optimise_ipeps(A, model, χ, params; Dz, ifWp=false)