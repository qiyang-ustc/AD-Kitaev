using OMEinsumContractionOrders

"""
    oc_H, oc_V = optcont(D::Int, χ::Int)
optimise the follow two einsum contractions for the given `D` and `χ` which are used to calculate the energy of the 2-site hamiltonian:
```
                                            a ────┬──── c          
a ────┬──c ──┬──── f                        │     b     │  
│     b      e     │                        ├─ e ─┼─ f ─┤  
├─ g ─┼─  h ─┼─ i ─┤                        g     h     i 
│     k      n     │                        ├─ j ─┼─ k ─┤ 
j ────┴──l ──┴──── o                        │     m     │ 
                                            l ────┴──── n 
```
where the central two block are six order tensor have extra bond `pq` and `rs`
"""
function optcont(D::Int, χ::Int)
    sd = Dict('a' => χ, 'b' => D^2,'c' => χ, 'e' => D^2, 'f' => χ, 'g' => D^2, 'h' => D^2, 'i' => D^2, 'j' => χ, 'k' => D^2, 'l' => χ, 'n' => D^2, 'o' => χ, 'p' => 2, 'q' => 2, 'r' => 2, 's' => 2)
    # for seed =20:100
    seed = 60
	Random.seed!(seed)
	# oc_H = optimize_code(ein"agj,abc,gkhbpq,jkl,fio,cef,hniers,lno -> pqrs", sd, TreeSA())
    oc_H = ein"(((agj,abc),gkhbpq),jkl),(((fio,cef),hniers),lno) -> pqrs"
	print("Horizontal Contraction Complexity(seed=$(seed))",OMEinsum.timespace_complexity(oc_H,sd),"\n")
    
    sd = Dict('a' => χ, 'b' => D^2, 'c' => χ, 'e' => D^2, 'f' => D^2, 'g' => χ, 'h' => D^2, 'i' => χ, 'j' => D^2, 'k' => D^2, 'l' => χ, 'm' => D^2, 'n' => χ, 'r' => 2, 's' => 2, 'p' => 2, 'q' => 2)
    # oc_V = optimize_code(ein"abc,aeg,ehfbpq,cfi,gjl,jmkhrs,ikn,lmn -> pqrs", sd, TreeSA())
    oc_V = ein"(((abc,aeg),ehfbpq),cfi),(gjl,(jmkhrs,(ikn,lmn))) -> pqrs"
    print("Vertical Contraction Complexity(seed=$(seed))",OMEinsum.timespace_complexity(oc_V,sd),"\n") 
    oc_H, oc_V
end


function expectation_value(h, ap, env, oc, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    # χ = size(ACu[1], 1)
    # ACu = map(j->reshape(j, (χ, Int(prod(size(j))/χ/χ), χ)), ACu)
    # ARu = map(j->reshape(j, (χ, Int(prod(size(j))/χ/χ), χ)), ARu)
    # ACd = map(j->reshape(j, (χ, Int(prod(size(j))/χ/χ), χ)), ACd)
    # ARd = map(j->reshape(j, (χ, Int(prod(size(j))/χ/χ), χ)), ARd)
    # FLu = map(j->reshape(j, (χ, Int(prod(size(j))/χ/χ), χ)), FLu)
    # FRu = map(j->reshape(j, (χ, Int(prod(size(j))/χ/χ), χ)), FRu)
    # FLo = map(j->reshape(j, (χ, Int(prod(size(j))/χ/χ), χ)), FLo)
    # FRo = map(j->reshape(j, (χ, Int(prod(size(j))/χ/χ), χ)), FRo)

    Ni, Nj = size(ap)
    oc_H, oc_V = oc
    etol = 0
    for j = 1:Nj, i = 1:1
        params.verbosity >= 4 && println("===========$i,$j===========")
        if i !== j
            h_H = h[1]
            h_V = h[3]
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni) 
            lr = oc_V(ACu[i,j],FLu[i,j],ap[i,j],FRu[i,j],FLo[ir,j],ap[ir,j],FRo[ir,j],conj(ACd[irr,j]))
            e = sum(ein"pqrs, pqrs -> "(lr,h_V))
            n = sum(ein"pprr -> "(lr))
            params.verbosity >= 4 && println("Vertical energy = $(e/n)")
            etol += e/n
        else
            h_H = h[2]
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        lr = oc_H(FLo[i,j],ACu[i,j],ap[i,j],conj(ACd[ir,j]),FRo[i,jr],ARu[i,jr],ap[i,jr],conj(ARd[ir,jr]))
        e = sum(ein"pqrs, pqrs -> "(lr,h_H))
        n = sum(ein"pprr -> "(lr))
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n
    end

    params.verbosity >= 3 && println("energy = $(etol/Ni/Nj)")
    return etol/Ni/Nj*2
end


"""
```
┌── Au─       ┌──        a──┬──b
c   │  =   λc c             c      
└── Ad─       └──        d──┴──e                
```
"""
function cmap(ci, Aui, Adi)
    cij = ein"(adi,acbi),dcei->bei"(ci,Aui,Adi)
    circshift(cij, (0,0,1))
end

function cint(A)
    χ, Ni, Nj = size(A)[[1,4,5]]
    atype = _arraytype(A)
    c = atype == Array ? rand(ComplexF64, χ, χ, Ni, Nj) : CUDA.rand(ComplexF64, χ, χ, Ni, Nj)
    return c
end

function cor_len(Au, Ad, c = cint(Au); kwargs...) 
    Ni,Nj = size(Au)[[4,5]]
    λc = zeros(eltype(c),Ni)
    ξ = 0.0
    for i in 1:Ni
        λcs, cs, info = eigsolve(X->cmap(X, Au[:,:,:,i,:], Ad[:,:,:,i,:]), c[:,:,i,:], 2, :LM; maxiter=100, ishermitian = false)
        info.converged == 0 && @warn "cor_len not converged"
        ξ = -1/log(abs(λcs[2]/λcs[1]))
    end
    return ξ
end

function observable(model, fdirection, field, type, folder, atype, D, χ, targχ, tol, maxiter, miniter, Ni, Nj; ifload = false)
    if field == 0.0
        observable_log = folder*"$(Ni)x$(Nj)/$(model)/D$(D)_χ$(targχ)_observable.log"
    else
        observable_log = folder*"$(Ni)x$(Nj)/$(model)_field$(fdirection)_$(@sprintf("%0.2f", field))$(type)/D$(D)_χ$(targχ)_observable.log"
    end
    if isfile(observable_log) && ifload
        println("load observable from $(observable_log)")
        f = open(observable_log, "r" )
        mag, ferro, stripy, zigzag, Neel, etol, ΔE, Cross = parse.(Float64,split(readline(f), "   "))
    else
        bulk, key = init_ipeps(model, fdirection, field; folder = folder, type = type, atype = atype, Ni = Ni, Nj = Nj, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
        folder, model, field, atype, Ni, Nj, D, χ, tol, maxiter, miniter = key
        h = hamiltonian(model)
        d = Int(2*model.S + 1) ^ 2
        oc = optcont(D, targχ, d)
        bulk = buildbcipeps(bulk,Ni,Nj)
        ap = [ein"abcdx,ijkly -> aibjckdlxy"(bulk[i], conj(bulk[i])) for i = 1:Ni*Nj]
        ap = [atype(reshape(ap[i], D^2, D^2, D^2, D^2, d, d)) for i = 1:Ni*Nj]
        ap = reshape(ap, Ni, Nj)
        a = atype(zeros(ComplexF64, D^2,D^2,D^2,D^2,Ni,Nj))
        for j in 1:Nj, i in 1:Ni
            a[:,:,:,:,i,j] = ein"ijklaa -> ijkl"(ap[i,j])
        end

        chkp_file_obs = folder*"obs_D$(D^2)_χ$(targχ).jld2"
        FL, FR = load(chkp_file_obs)["env"]
        chkp_file_up = folder*"up_D$(D^2)_χ$(targχ).jld2"                     
        rtup = SquareVUMPSRuntime(a, chkp_file_up, targχ; verbose = false)   
        FLu, FRu, ALu, ARu, Cu = rtup.FL, rtup.FR, rtup.AL, rtup.AR, rtup.C
        chkp_file_down = folder*"down_D$(D^2)_χ$(targχ).jld2"                              
        rtdown = SquareVUMPSRuntime(a, chkp_file_down, targχ; verbose = false)   
        ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C
        ACu = ALCtoAC(ALu, Cu)
        ACd = ALCtoAC(ALd, Cd)

        ALu, Cu, ACu, ARu, ALd, Cd, ACd, ARd, FL, FR, FLu, FRu = map(atype, [ALu, Cu, ACu, ARu, ALd, Cd, ACd, ARd, FL, FR, FLu, FRu])
        
        ξ = cor_len(ALu, ALd)

        M = Array{Array{ComplexF64,1},3}(undef, Ni, Nj, 2)
        Sx = const_Sx(model.S)
        Sy = const_Sy(model.S)
        Sz = const_Sz(model.S)
        Sx1 = reshape(ein"ab,cd -> acbd"(Sx, I(Int(sqrt(d)))), (d,d))
        Sx2 = reshape(ein"ab,cd -> acbd"(I(Int(sqrt(d))), Sx), (d,d))
        Sy1 = reshape(ein"ab,cd -> acbd"(Sy, I(Int(sqrt(d)))), (d,d))
        Sy2 = reshape(ein"ab,cd -> acbd"(I(Int(sqrt(d))), Sy), (d,d))
        Sz1 = reshape(ein"ab,cd -> acbd"(Sz, I(Int(sqrt(d)))), (d,d))
        Sz2 = reshape(ein"ab,cd -> acbd"(I(Int(sqrt(d))), Sz), (d,d))
        etol = 0.0
        logfile = open(observable_log, "a")
        for j = 1:Nj, i = 1:Ni
            jr = j + 1 - (j==Nj)*Nj
            ir = Ni + 1 - i
            lr3 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL[:,:,:,i,j],ACu[:,:,:,i,j],ap[i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,j])
            Mx1 = ein"pq, pq -> "(Array(lr3),Sx1)
            Mx2 = ein"pq, pq -> "(Array(lr3),Sx2)
            My1 = ein"pq, pq -> "(Array(lr3),Sy1)
            My2 = ein"pq, pq -> "(Array(lr3),Sy2)
            Mz1 = ein"pq, pq -> "(Array(lr3),Sz1)
            Mz2 = ein"pq, pq -> "(Array(lr3),Sz2)
            n3 = ein"pp -> "(lr3)
            M[i,j,1] = [Array(Mx1)[]/Array(n3)[], Array(My1)[]/Array(n3)[], Array(Mz1)[]/Array(n3)[]]
            M[i,j,2] = [Array(Mx2)[]/Array(n3)[], Array(My2)[]/Array(n3)[], Array(Mz2)[]/Array(n3)[]]
            print("M[[$(i),$(j),$(1)]] = {")
            for k = 1:3 
                print(real(M[i,j,1][k])) 
                k == 3 ? println("};") : print(",")
            end
            print("M[[$(i),$(j),$(2)]] = {")
            for k = 1:3 
                print(real(M[i,j,2][k])) 
                k == 3 ? println("};") : print(",")
            end
            if field != 0.0
                etol -= (real(M[i,j,1] + M[i,j,2]))' * field / 2
            end
            message = "M[[$(i),$(j),$(1)]] = $(M[i,j,1])\nM[[$(i),$(j),$(2)]] = $(M[i,j,2])\n"
            write(logfile, message)
        end

        oc1, oc2 = oc
        hx, hy, hz = h
        Id = I(Int(sqrt(d)))
        Sx = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(Id, hx, Id), (d,d,d,d)))
        Sy = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(Id, hy, Id), (d,d,d,d)))
        Sz = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(Id, hz, Id), (d,d,d,d)))
        ap /= norm(ap)
        hx = atype(reshape(permutedims(hx, (1,3,2,4)), (d,d)))
        hy = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(Id, hy, Id), (d,d,d,d)))
        hz = atype(reshape(ein"ae,bfcg,dh -> abefcdgh"(Id, hz, Id), (d,d,d,d)))

        Ex, Ey, Ez = 0, 0, 0
        for j = 1:Nj, i = 1:Ni
            println("===========$i,$j===========")
            ir = Ni + 1 - i
            jr = j + 1 - (j==Nj) * Nj
            lr = oc1(FL[:,:,:,i,j],ACu[:,:,:,i,j],ap[i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,jr],ARu[:,:,:,i,jr],ap[i,jr],ARd[:,:,:,ir,jr])
            e = Array(ein"pqrs, pqrs -> "(lr,hz))[]
            n =  Array(ein"pprr -> "(lr))[]
            println("xx = $(Array(ein"pqrs, pqrs -> "(lr,Sx))[]/n)")
            println("yy = $(Array(ein"pqrs, pqrs -> "(lr,Sy))[]/n)")
            println("zz = $(Array(ein"pqrs, pqrs -> "(lr,Sz))[]/n)")
            println("hz = $(e/n)")
            Ez   += e/n
            etol += e/n

            lr = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL[:,:,:,i,j],ACu[:,:,:,i,j],ap[i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,j])
            e = Array(ein"pq, pq -> "(lr,hx))[]
            n = Array(ein"pp -> "(lr))[]
            println("hx = $(e/n)")
            Ex   += e/n
            etol += e/n

            ir  =  i + 1 - (i==Ni) * Ni
            irr = Ni - i + (i==Ni) * Ni
            lr = oc2(ACu[:,:,:,i,j],FLu[:,:,:,i,j],ap[i,j],FRu[:,:,:,i,j],FL[:,:,:,ir,j],ap[ir,j],FR[:,:,:,ir,j],ACd[:,:,:,irr,j])
            e = Array(ein"pqrs, pqrs -> "(lr,hy))[]
            n =  Array(ein"pprr -> "(lr))[]
            println("hy = $(e/n)")
            Ey   += e/n
            etol += e/n
        end
        println("e = $(etol/Ni/Nj)")
        etol = real(etol/(Ni * Nj))
        # ΔE = real(Ex - (Ey + Ez)/2)

        message = "E     = $(etol)\nEx    = $(Ex)\nEy    = $(Ey)\nEz    = $(Ez)\nξ    = $(ξ)\n"
        write(logfile, message)
        close(logfile)
    end
    return etol, Ex, Ey, Ez, M
end

function fidelity(key1,key2)
    model1, fdirection1, field1, type1, folder1, D1, χ1, tol1, maxiter1, miniter1 = key1
    model2, fdirection2, field2, type2, folder2, D2, χ2, tol2, maxiter2, miniter2 = key2
    if field == 0.0
        fidelity_log = folder*"$(model1)/D$(D1)_χ$(χ1)_fidelity.log"
    else
        fidelity_log = folder*"$(model1)_field$(fdirection1)_$(field1)$(type1)/D$(D1)_χ$(χ1)_fidelity.log"
    end
    # if isfile(fidelity_log)
    #     println("load fidelity from $(fidelity_log)")
    #     f = open(fidelity_log, "r" )
    #     Ftol = parse.(Float64,readline(f))
    # else
        bulk1, key1 = init_ipeps(model1, fdirection1, field1; folder = folder1, type = type1, atype = CuArray, D=D1, χ=χ1, tol=tol1, maxiter=maxiter1, miniter=miniter1, verbose = true)
        folder1, model1, field1, atype1, D1, χ1, tol1, maxiter1, miniter1 = key1
        Ni = 1
        Nj = Int(size(bulk1,6) / Ni)
        bulk1 = buildbcipeps(bulk1,Ni,Nj)
        ap1 = [ein"abcdx,ijkly -> aibjckdlxy"(bulk1[i], conj(bulk1[i])) for i = 1:Ni*Nj]
        ap1 = [reshape(ap1[i], D^2, D^2, D^2, D^2, 4, 4) for i = 1:Ni*Nj]
        ap1 = reshape(ap1, Ni, Nj)
        a1 = [ein"ijklaa -> ijkl"(ap1[i]) for i = 1:Ni*Nj]
        a1 = reshape(a1, Ni, Nj)
        
        chkp_file_obs1 = folder1*"obs_D$(D1^2)_chi$(χ1).jld2"
        FL1, FR1 = load(chkp_file_obs1)["env"]
        chkp_file_up1 = folder1*"up_D$(D1^2)_χ$(χ1).jld2"                     
        rtup1 = SquareBCVUMPSRuntime(a1, chkp_file_up1, χ1; verbose = false)   
        ALu1, Cu1 = rtup1.AL, rtup1.C
        chkp_file_down1 = folder1*"down_D$(D1^2)_χ$(χ1).jld2"                              
        rtdown1 = SquareBCVUMPSRuntime(a1, chkp_file_down1, χ1; verbose = false)   
        ALd1,Cd1 = rtdown1.AL,rtdown1.C
        ACu1 = ALCtoAC(ALu1, Cu1)
        ACd1 = ALCtoAC(ALd1, Cd1) 

        bulk2, key2 = init_ipeps(model2, fdirection2, field2; folder = folder2, type = type2, atype = CuArray, D=D2, χ=χ2, tol=tol2, maxiter=maxiter2, miniter=miniter2, verbose = true)
        folder2, model2, field2, atype2, D2, χ2, tol2, maxiter2, miniter2 = key2
        bulk2 = buildbcipeps(bulk2,Ni,Nj)
        ap2 = [ein"abcdx,ijkly -> aibjckdlxy"(bulk2[i], conj(bulk2[i])) for i = 1:Ni*Nj]
        ap2 = [reshape(ap2[i], D^2, D^2, D^2, D^2, 4, 4) for i = 1:Ni*Nj]
        ap2 = reshape(ap2, Ni, Nj)
        a2 = [ein"ijklaa -> ijkl"(ap2[i]) for i = 1:Ni*Nj]
        a2 = reshape(a2, Ni, Nj)

        chkp_file_obs2 = folder2*"obs_D$(D2^2)_chi$(χ2).jld2"
        FL2, FR2 = load(chkp_file_obs2)["env"]
        chkp_file_up2 = folder2*"up_D$(D2^2)_χ$(χ2).jld2"
        rtup2 = SquareBCVUMPSRuntime(a2, chkp_file_up2, χ2; verbose = false)
        ALu2, Cu2 = rtup2.AL, rtup2.C
        chkp_file_down2 = folder2*"down_D$(D2^2)_χ$(χ2).jld2"
        rtdown2 = SquareBCVUMPSRuntime(a2, chkp_file_down2, χ2; verbose = false)
        ALd2, Cd2 = rtdown2.AL, rtdown2.C
        ACu2 = ALCtoAC(ALu2, Cu2)
        ACd2 = ALCtoAC(ALd2, Cd2)

        Ftol = 1
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            lr1 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL2[i,j],ACu2[i,j],ap2[i,j],ACd2[ir,j],FR2[i,j])
            n1 = ein"pp -> "(lr1)
            ρ = Array(lr1) / Array(n1)[]
            lr2 = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FR1[i,j],ACu1[i,j],ap1[i,j],ACd1[ir,j],FL1[i,j])
            F = ein"pq,qp ->"(lr2,ρ)
            n2 = ein"pp -> "(lr2)
            Ftol *= Array(F)[]/Array(n2)[]
            @show Ftol
        end
        Ftol = sqrt(norm(Ftol))
        message = "$(Ftol)\n"
        logfile = open(fidelity_log, "w")
        write(logfile, message)
        close(logfile)
    # end
    return Ftol
end
