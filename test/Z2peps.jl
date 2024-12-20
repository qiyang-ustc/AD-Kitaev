using Test
using Random
using OMEinsum
import Base: +, -, *, /
using LinearAlgebra

begin
    function diag_front2(T, D)
        T′ = zeros(D,D,prod(size(T))÷D^2)
        T = reshape(T,D,D,prod(size(T))÷D^2)
        for i in 1:D, j in 1:D, k in 1:prod(size(T))÷D^2
            if i == j
                T′[i,j,k] = T[i,j,k]
            elseif i < j
                T′[i,j,k] = (T[i,j,k] + T[j,i,k])/sqrt(2)
            else
                T′[i,j,k] = (T[i,j,k] - T[j,i,k])/sqrt(2) 
            end
        end

        return reshape(T′,D^2,D^2)
    end
    
    struct Z2peps
        real::Array
        imag::Array
        function Z2peps(A)
            new(real(A), imag(A))
        end
    
        function Z2peps(real, imag)
            new(real, imag)
        end
    end
    
    *(Z1::Z2peps, Z2::Z2peps) = Z2peps(Z1.real * Z2.real - Z1.imag * Z2.imag, Z1.real * Z2.imag + Z1.imag * Z2.real)
    back(x::Z2peps) = x.real + 1im * x.imag
end

@testset "Z2 real peps" begin
    Random.seed!(42) 
    D = 2
    d = 1
    A = rand(D,d,D)
    T = reshape(ein"abc,dbe->adce"(A,A),D^2,D^2)
    # T = rand(D^2,D^2)


    T′ = diag_front2(T, D)
    T′ = diag_front2(T′', D)'
    λ1, _ = eigen(T)
    λ2, _ = eigen(T′)
    @test λ1 ≈ λ2
    @test sum(T′.*T′) ≈  sum(T.*T)

    @show T′ reshape(T′,D,D,D,D)[2,1,2,1]
end


@testset "Z2 complex peps" begin
    Random.seed!(42) 
    D = 2
    d = 1
    A = rand(ComplexF64, D,d,D)
    T = reshape(ein"abc,dbe->adce"(A,conj(A)),D^2,D^2)
    
    T = real(T)
    T′ = diag_front2(T, D)
    T′ = diag_front2(T′', D)'
    @show T T′

    λ1, _ = eigen(T)
    λ2, _ = eigen(T′)
    @test λ1 ≈ λ2
    @test sum(T′.*T′) ≈  sum(T.*T)

    T = imag(T)
    T′ = diag_front2(T, D)
    T′ = diag_front2(T′', D)'
    @show T T′

    λ1, _ = eigen(T)
    λ2, _ = eigen(T′)
    @test sum(λ1) ≈ sum(λ2)
    @test sum(T′.*T′) ≈  sum(T.*T)
end