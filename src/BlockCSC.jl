import Base: +, *
import SparseArrays: nnz

import LinearAlgebra: mul!

using StaticArrays


"""
    {D,T,L}(mat_of_mats)

A BlockCSC is a sparse matrix of square static matrices (StaticArray) that all have the same size.
It can be constructed from a matrix of matrices, like

    BlockCSC([randn(2,2) for i in 1:2, j in 1:2])

Note that it is NOT the same as a BlockMatrix, as defined in BlockArrays.
"""
struct BlockCSC{D,T,L} <: AbstractArray{T,2} 
    mat::SparseMatrixCSC{StaticArrays.SArray{Tuple{D,D},T,2,L}, Int}
end

function BlockCSC(s::AbstractSparseArray{StaticArrays.SArray{Tuple{D,D},T,2,L},I,N} where N where L where I) where T where D
    return BlockCSC{D,T,L}(s)
end

function BlockCSC(s::AbstractSparseArray{Array{T,2},I,2} where I) where T
    si,sj,sv = findnz(s)
    d = size(sv[1],1)
    return BlockCSC(sparse(si,sj,map(SMatrix{d,d},sv)))
end


function BlockCSC(a::DenseArray{Array{T,2},2} where T)
    BlockCSC(sparse(a))
end

function BlockCSC(a::Array{StaticArrays.SArray{Tuple{D,D},T,N,L},2} where D where N where L where T)
    BlockCSC(sparse(a))
end

function BlockCSC(a::SparseMatrixCSC{Tv,Ti}) where {Ti, Tv <: Real}
    BlockCSC(map(SMatrix{1,1}, a))
end

Base.size(a::BlockCSC) = size(a.mat)

Base.getindex(a::BlockCSC, i, j) = a.mat[i,j]

#=
The following does not work because of a Julia bug, documented here:

=#
#Base.setindex!(a::BlockCSC, v, i, j) = a.mat[i,j] = v

# the following comes from https://github.com/JuliaLang/julia/blob/dc79628272ab209e8bc9ab796d75587dcefc0fb4/stdlib/SparseArrays/src/sparsematrix.jl
function BlockCSC_setindex_scalar!(A::SparseMatrixCSC{Tv,Ti}, _v, _i::Integer, _j::Integer) where {Tv,Ti<:Integer}
    v = convert(Tv, _v)
    i = convert(Ti, _i)
    j = convert(Ti, _j)
    if !((1 <= i <= A.m) & (1 <= j <= A.n))
        throw(BoundsError(A, (i,j)))
    end
    coljfirstk = Int(A.colptr[j])
    coljlastk = Int(A.colptr[j+1] - 1)
    searchk = searchsortedfirst(A.rowval, i, coljfirstk, coljlastk, Base.Order.Forward)
    if searchk <= coljlastk && A.rowval[searchk] == i
        # Column j contains entry A[i,j]. Update and return
        A.nzval[searchk] = v
        return A
    end
    # Column j does not contain entry A[i,j]. If v is nonzero, insert entry A[i,j] = v
    # and return. If to the contrary v is zero, then simply return.
    if v != 0
        insert!(A.rowval, searchk, i)
        insert!(A.nzval, searchk, v)
        @simd for m in (j + 1):(A.n + 1)
            @inbounds A.colptr[m] += 1
        end
    end
    return A
end

function Base.setindex!(a::BlockCSC, v, i, j)
    BlockCSC_setindex_scalar!(a.mat, v, i, j)

end


function +(C1::BlockCSC, C2::BlockCSC)
    BlockCSC(C1.mat + C2.mat)
end

eye(a::BlockCSC) = BlockCSC(eye(a.mat))

SparseArrays.nnz(a::BlockCSC) = nnz(a.mat)

# the following allows us to multiply sparse matrices of static arrays by vectors of static arrays

mul(As::AbstractSparseArray{StaticArrays.SArray{Tuple{D,D},T,2,L}},
     x::AbstractArray{SVector{D, T}}) where {D, T, L} = mul!(similar(x), As, x, one(T), zero(T))


mul!(y::AbstractArray{SArray{Tuple{D},T,1,D},1}, B::BlockCSC{D,T,L}, 
    x::AbstractArray{SArray{Tuple{D},T,1,D},1}) where {D, T, L} = mul!(y, B.mat, x, one(T), zero(T))




*(B::BlockCSC{D,T,L}, 
    v::AbstractArray{SArray{Tuple{D},T,1,D},1}) where {D,T,L} = mul(B.mat, v)

function *(B::BlockCSC{D,T,L}, x::Vector{T}) where {D,T,L} 
    xs = reinterpret(SVector{D,T},x)
    c = mul!(similar(xs), B.mat, xs, one(T), zero(T))
    return collect(reinterpret(T, c))
end



import Laplacians.isConnected
isConnected(a::BlockCSC) = isConnected(a.mat)

"""
    S = expandBlockCSC(M::BlockCSC)

expand M into a regular sparse matrix S.
"""
function expandBlockCSC(a::BlockCSC{D,T,L})  where {D,T,L}

    d = D
    d2 = d*d

    n = size(a,1)*d

    m = nnz(a.mat)

    (ai,aj,av) = findnz(a.mat)

    si = kron(d*(ai.-1),ones(Int,d2)) + kron(ones(Int,m*d), 1:d)
    sj = kron(d*(aj.-1),ones(Int,d2)) + kron(ones(Int,m), kron(1:d,ones(Int,d)))
    
    sv = BlockCSC_nonzeros(eltype(a),av,d,m)
 
    ret = sparse(si,sj,sv,n,n)

    return ret
end

function BlockCSC_nonzeros(elt, av, d, m)
    tv = Vector{elt}(undef,d*d*m)
    d2 = d*d
    r0 = 1
    r1 = d2
    for i in 1:length(av)
        tv[r0:r1] .= vec(av[i])
        r0 += d2
        r1 += d2
    end
    return tv    
end


"""
    at = transpose(a::BlockCSC)

Compute the transpose, doing it recursively.

```jldoctest
julia> a = BlockCSC([randn(2,2) for i in 1:2, j in 1:2]);
julia> iszero(expandBlockCSC(a)' - expandBlockCSC(transpose(a)))
true
```
"""
function Base.transpose(a::BlockCSC)
    matt = sparse(transpose(a.mat))

    return BlockCSC(matt)
end

"""
    vs = BlockVec(D, v)

View a vector v as a vector of D-dimensional StaticArrays.
Note that this returns a *view*, so the data is unchanged.
You may wish to `copy` it.  The opposite is FlatVec.
"""
BlockVec(D, v::AbstractArray{T}) where {T <: Real} = 
    reinterpret(SVector{D,T},v)

"""
    v = FlatVec(vs)

View a vector of StaticArrays as a flat vector.
Note that this returns a *view*, so the data is unchanged.
You may wish to `copy` it.  The opposite is BlockVec.
"""
FlatVec(v::Vector{SVector{D,T}}) where {D,T} = reinterpret(T, v)


import SparseArrays.sparse

sparse(M::BlockCSC) = expandBlockCSC(M)
