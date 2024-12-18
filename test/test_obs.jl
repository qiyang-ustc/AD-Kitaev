using AD_Kitaev
using Random
using CUDA
using TeneT

#####################################    parameters      ###################################
Random.seed!(42)
atype = Array
Ni, Nj = 2, 2
D, χ = 2, 10
No = 92
S = 1.0
ifWp = false
model = Kitaev(S,1.0,1.0,1.0)
Dz = 0.0
method = :brickwall
# method = :merge
folder = "data/$method/Dz$Dz/$model/$(Ni)x$(Nj)/D$(D)_χ$(χ)/"
############################################################################################

if method == :merge
    d = Int(2*S+1)^2
elseif method == :brickwall
    d = Int(2*S+1)
end
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false, 
                     ifsimple_eig=true,
                     maxiter=30, 
                     miniter=1,
                     maxiter_ad=0,
                     miniter_ad=0,
                     verbosity=2
)
params = iPEPSOptimize{method}(boundary_alg=boundary_alg,
                               reuse_env=true, 
                               verbosity=4, 
                               maxiter=1000,
                               tol=1e-10,
                               folder=folder
)
# A = AD_Kitaev.init_ipeps_form_small_spin(;atype, D=D, file=file, d=Int(2*S+1)^2, Ni=Ni, Nj=Nj)
A = AD_Kitaev.init_ipeps(;atype, params, No, D, d, Ni, Nj)

e, mag = observable(A, model, Dz, χ, params; ifWp)
# @show e, mag