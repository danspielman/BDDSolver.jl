
#=======================================================================

We now use the PermLTCSC structure to solve systems.
The other code is just for testing and recording other things we tried.

=========================================================================#
#=
function forwardsolve!(lt::PermLTCSC{Ti,StaticArrays.SArray{Tuple{D,D},T,2,L}},
    b::AbstractArray{SVector{D, T}}) where {Ti, D, T, L}
    n = length(lt.colptr)-1

    for j in 1:n
        col = lt.rowval[lt.colptr[j]]

        b[col] = lt.nzval[lt.colptr[j]] \ b[col]

        for ii in (lt.colptr[j]+1):(lt.colptr[j+1]-1)
            i = lt.rowval[ii]
            b[i] -= lt.nzval[ii] * b[col]
        end
    end
end

function forwardsolve!(lt::PermLTCSC{Ti,Tv}, b::Vector{Tv}) where {Ti,Tv}
    n = length(lt.colptr)-1

    @inbounds for j in 1:n
        col = lt.rowval[lt.colptr[j]]

        b[col] /= lt.nzval[lt.colptr[j]]

        for ii in (lt.colptr[j]+1):(lt.colptr[j+1]-1)
            i = lt.rowval[ii]
            b[i] -= b[col] * lt.nzval[ii]
        end
    end
end
=#

import Laplacians.pcg

function forwardsolve!(lt::PermLTCSC, b)
    n = length(lt.colptr)-1

    @inbounds for j in 1:n
        col = lt.rowval[lt.colptr[j]]

        b[col] = lt.nzval[lt.colptr[j]] \ b[col]

        for ii in (lt.colptr[j]+1):(lt.colptr[j+1]-1)
            i = lt.rowval[ii]
            b[i] -= lt.nzval[ii] * b[col]
        end
    end
end

# this works only when L has been permuted to lower tri.
function forwardsolve!(L::LTCSC, b::Vector)

    n = length(b)

    @inbounds for j in 1:n

        b[j] = L.nzval[L.colptr[j]] \ b[j]

        for ii in (L.colptr[j]+1):(L.colptr[j+1]-1)
            i = L.rowval[ii]
            b[i] -= L.nzval[ii] * b[j]
        end
    end

end


function backsolve!(lt::PermLTCSC, b)
    n = length(lt.colptr)-1

    @inbounds for j in n:-1:1
        col = lt.rowval[lt.colptr[j]]

        for ii in (lt.colptr[j]+1):(lt.colptr[j+1]-1)
            b[col] -= lt.nzval[ii]' * b[lt.rowval[ii]]
        end
        b[col] = lt.nzval[lt.colptr[j]] \ b[col]
    end
end

# this works only when L has been permuted to lower tri.
function backsolve!(L::LTCSC, b::Vector)
    n = length(b)

    @inbounds for j in n:-1:1
        for ii in (L.colptr[j]+1):(L.colptr[j+1]-1)
            b[j] -= L.nzval[ii]' * b[L.rowval[ii]]
        end
        b[j] = L.nzval[L.colptr[j]] \ b[j]
    end
end


# For vectors y of block vectors, and block matrix lt.
function solver!(lt::AbsLTCSC{Ti,StaticArrays.SArray{Tuple{D,D},T,2,L}},
    y::AbstractArray{SVector{D, T}}) where {Ti, D, T, L}

    n = length(lt.colptr)-1

    @assert length(y) == n

    forwardsolve!(lt, y)
    @inbounds for i in 1:length(y)
        y[lt.rowval[lt.colptr[i]]] = lt.nzval[lt.colptr[i]] * y[lt.rowval[lt.colptr[i]]]
    end
    backsolve!(lt, y)
end


# for real entry  lt, and real vectors
function solver!(lt::AbsLTCSC{Ti,Tv},
    y::Vector{Tv}) where {Ti, Tv}

    n = length(lt.colptr)-1

    @assert length(y) == n

    forwardsolve!(lt, y)
    @inbounds for i in 1:length(y)
        y[lt.rowval[lt.colptr[i]]] = lt.nzval[lt.colptr[i]] * y[lt.rowval[lt.colptr[i]]]
    end
    backsolve!(lt, y)
end

# For vectors y of block vectors, and block matrix lt.
function solver(lt::AbsLTCSC{Ti,StaticArrays.SArray{Tuple{D,D},T,2,L}},
    b::AbstractArray{SVector{D, T}}) where {Ti, D, T, L}

    y = copy(b)
    solver!(lt, y)
    return y
end

# For vectors y of real entries, and block matrix lt.
function solver(lt::AbsLTCSC{Ti,StaticArrays.SArray{Tuple{D,D},T,2,L}},
    b::Vector{T}) where {Ti, D, T, L}
    y = copy(reinterpret(SVector{D,T},b))
    solver!(lt, y)
    return collect(reinterpret(T,y))
end

# for real entry  lt, and real vectors
function solver(lt::AbsLTCSC{Ti,Tv},
    b::Vector{Tv}) where {Ti, Tv}
    y = copy(b)
    solver!(lt, y)
    return y
end

function pcg(mat::BlockCSC{D,Tv,L}, b::Vector{Tv}, pre::Function;
    tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[],
    stag_test::Integer=0)::Vector{Tv} where {D,Tv,L}

    FlatVec(
        pcg(mat, BlockVec(D,b), pre,
        tol=tol, maxits=maxits, maxtime=maxtime,
        pcgIts=pcgIts,
        verbose=verbose,
        stag_test = stag_test)
     )

end

function pcg_inplace(mat::BlockCSC{D,Tv,L}, b::Vector{Tv}, pre::Function;
    tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[],
    stag_test::Integer=0)::Vector{Tv} where {D,Tv,L}

    FlatVec(
        pcg_inplace(mat, BlockVec(D,b), pre,
        tol=tol, maxits=maxits, maxtime=maxtime,
        pcgIts=pcgIts,
        verbose=verbose,
        stag_test = stag_test)
     )

end


#import Laplacians.pcg
import Laplacians.axpy2!


function pcg(mat::BlockCSC{D,Tv,L},
    b::AbstractVector{SArray{Tuple{D},Tv,1,D}},
    pre::Function;
    tol::Real=1e-6,
    maxits=Inf,
    maxtime=Inf,
    verbose=false,
    pcgIts=Int[],
    stag_test::Integer=0) where {D,Tv,L}

    local al::Tv

    n = size(mat,2)

    nb = norm(b)

    # If input vector is zero, quit
    if nb == 0
    return zeros(size(b))
    end

    x = zeros(SVector{D,Tv} ,n)
    bestx = zeros(SVector{D,Tv}, n)
    bestnr = one(Tv)

    r = copy(b)
    z = pre(r)
    p = copy(z)

    rho = dot(r, z)
    best_rho = rho
    stag_count = 0

    t1 = time()

    itcnt = 0
    while itcnt < maxits
        itcnt = itcnt+1

        q = mat*p

        pq = dot(p,q)

        if (pq < eps(Tv) || isinf(pq))
            if verbose
                println("PCG Stopped due to small or large pq")
            end
            break
        end

        al = rho/pq

        # the following line could cause slowdown
        if al*norm(p) < eps(Tv)*norm(x)
            if verbose
                println("PCG: Stopped due to stagnation.")
            end
            break
        end

        axpy2!(al,p,x)
        # x = x + al * p
        #=
        @inbounds @simd for i in 1:n
            x[i] += al*p[i]
        end
        =#
        #axpy

        axpy2!(-al,q,r)
        #r .= r .- al.*q
        #=
        @inbounds @simd for i in 1:n
            r[i] -= al*q[i]
        end
        =#

        nr = norm(r)/nb
        if nr < bestnr
            bestnr = nr
            @inbounds @simd for i in 1:n
                bestx[i] = x[i]
            end
        end

        if nr < tol #Converged?
            break
        end

        # here is the top of the code in numerical templates

        z = pre(r)

        oldrho = rho
        rho = dot(z, r) # this is gamma in hypre.

        if rho < best_rho*(1-1/stag_test)
            best_rho = rho
            stag_count = 0
        else
            if stag_test > 0
                if best_rho > (1-1/stag_test)*rho
                    stag_count += 1
                    if stag_count > stag_test
                        println("PCG Stopped by stagnation test ", stag_test)
                        break
                    end
                end
            end
        end

        if (rho < eps(Tv) || isinf(rho))
            if verbose
                println("PCG Stopped due to small or large rho")
            end
            break
        end

        # the following would have higher accuracy
        #       rho = sum(r.^2)

        beta = rho/oldrho
        if (beta < eps(Tv) || isinf(beta))
            if verbose
                println("PCG Stopped due to small or large beta")
            end
            break
        end

        bzbeta!(beta,p,z)
        #=
        # p = z + beta*p
        @inbounds @simd for i in 1:n
            p[i] = z[i] + beta*p[i]
        end
        =#

        if (time() - t1) > maxtime
            if verbose
                println("PCG New stopped at maxtime.")
            end
            break
        end

    end

    if verbose
        println("PCG stopped after: ", round((time() - t1),digits=3), " seconds and ", itcnt, " iterations with relative error ", (norm(r)/norm(b)), ".")
    end

    if length(pcgIts) > 0
        pcgIts[1] = itcnt
    end


    return bestx
end

# uses a preconditioner that acts in place
function pcg_inplace(mat::BlockCSC{D,Tv,L},
    b::AbstractVector{SArray{Tuple{D},Tv,1,D}},
    pre!::Function;
    tol::Real=1e-6,
    maxits=Inf,
    maxtime=Inf,
    verbose=false,
    pcgIts=Int[],
    stag_test::Integer=0) where {D,Tv,L}

    local al::Tv

    n = size(mat,2)

    nb = norm(b)

    # If input vector is zero, quit
    if nb == 0
    return zeros(size(b))
    end

    x = zeros(SVector{D,Tv} ,n)
    bestx = zeros(SVector{D,Tv}, n)
    bestnr = one(Tv)

    r = copy(b)
    z = copy(r)
    pre!(z)
    p = copy(z)
    q = similar(z)

    rho = dot(r, z)
    best_rho = rho
    stag_count = 0

    t1 = time()

    itcnt = 0
    while itcnt < maxits
        itcnt = itcnt+1

        mul!(q, mat, p)
        #q = mat*p

        pq = dot(p,q)

        if (pq < eps(Tv) || isinf(pq))
            if verbose
                println("PCG Stopped due to small or large pq")
            end
            break
        end

        al = rho/pq

        # the following line could cause slowdown
        if al*norm(p) < eps(Tv)*norm(x)
            if verbose
                println("PCG: Stopped due to stagnation.")
            end
            break
        end

        axpy2!(al,p,x)


        axpy2!(-al,q,r)


        nr = norm(r)/nb
        if nr < bestnr
            bestnr = nr
            @inbounds @simd for i in 1:n
                bestx[i] = x[i]
            end
        end

        if nr < tol #Converged?
            break
        end

        # here is the top of the code in numerical templates

        copy!(z,r) # copy r in to z
        pre!(z)

        oldrho = rho
        rho = dot(z, r) # this is gamma in hypre.

        if rho < best_rho*(1-1/stag_test)
            best_rho = rho
            stag_count = 0
        else
            if stag_test > 0
                if best_rho > (1-1/stag_test)*rho
                    stag_count += 1
                    if stag_count > stag_test
                        println("PCG Stopped by stagnation test ", stag_test)
                        break
                    end
                end
            end
        end

        if (rho < eps(Tv) || isinf(rho))
            if verbose
                println("PCG Stopped due to small or large rho")
            end
            break
        end

        # the following would have higher accuracy
        #       rho = sum(r.^2)

        beta = rho/oldrho
        if (beta < eps(Tv) || isinf(beta))
            if verbose
                println("PCG Stopped due to small or large beta")
            end
            break
        end

        bzbeta!(beta,p,z)
        #=
        # p = z + beta*p
        @inbounds @simd for i in 1:n
            p[i] = z[i] + beta*p[i]
        end
        =#

        if (time() - t1) > maxtime
            if verbose
                println("PCG New stopped at maxtime.")
            end
            break
        end

    end

    if verbose
        println("PCG stopped after: ", round((time() - t1),digits=3), " seconds and ", itcnt, " iterations with relative error ", (norm(r)/norm(b)), ".")
    end

    if length(pcgIts) > 0
        pcgIts[1] = itcnt
    end


    return bestx
end

function axpy2!(al,p::AbstractArray{SVector{D,T}},
        x::AbstractArray{SVector{D,T}}) where {D, T}
    n = length(x)
    @inbounds @simd for i in 1:n
        x[i] = x[i] + al*p[i]
    end
  end

  function bzbeta!(beta,p::AbstractArray{SVector{D,T}},
    z::AbstractArray{SVector{D,T}}) where {D, T}
    n = length(p)
    @inbounds @simd for i in 1:n
        p[i] = z[i] + beta*p[i]
    end
  end

#=============================================================

The routines that do the solve: forwards, backwards, and diagonal inversion.
Note that they have special versions for D=1, and probably should for D=2 and 3.

These all use the ops structure.  It should be replaced.

=============================================================#

mapD(D,i) = (D*(i-1)+1):(D*i)

# should hard-code D=2 and D=3
function forward!(ops, D, y)
    if D == 1
        forward1!(ops, y)
    else
        forwardD!(ops, D, y)
    end
end

# For D = 1
function forward1!(ops, y)

    for i in 1:length(ops)
        op = ops[i]

        y[op.j] += op.wji * y[op.i]
        y[op.i] = op.wii*y[op.i]
    end
end

#=
# This is the idealized function, but it is not fast enough.
=#
#=
function forwardD!(ops, D, y)

    for i in 1:length(ops)
        op = ops[i]
        di = mapD(D,op.i)
        dj = mapD(D,op.j)


        y[dj] = y[dj] + op.wji * y[di]
        y[di] = op.wii*y[di]
    end
end
=#



function forwardD!(ops, D, y)

    tmp = similar(y[1:D])

    @inbounds for i in 1:length(ops)
        op = ops[i]

        di = D*(op.i-1)
        dj = D*(op.j-1)

        # The following lines simulate:
        # y[dj] = y[dj] + op.wji * y[di]

        for j in 1:D
            for k in 1:D
                y[dj+j] += op.wji[j,k] * y[di+k]
            end
        end

        # The following lines simulate:
        # y[di] = op.wii*y[di]

        tmp .= 0.0

        for j in 1:D
            for k in 1:D
                tmp[j] += op.wii[j,k] * y[di+k]
            end
        end

        for j in 1:D
            y[di+j] = tmp[j]
        end
    end
end


#=
Invert diag d[i] on ith block of y, for each i.
Ideal routine, which is too slow, is:

    for i in 1:(length(d))

        if is_non_zero(d[i])
            di = mapD(D,i)
            y[di] = inv(d[i])*y[di]
        end

    end

=#
function diag!(d, D, y)
    if D == 1
        diag1!(d, y)
    else
        diagD!(d, D, y)
    end
end

function diag1!(d, y)
    for i in 1:length(d)
        y[i] *= d[i]
    end
end

function diagD!(d, D, y)

    tmp = similar(y[1:D])

    for i in 1:length(d)
        Di = D*(i-1)
        y[(Di+1):(Di+D)] = d[i]*y[(Di+1):(Di+D)]
    end
end

#=
You would think we should use the following, but it is slower
=#
function diagD2!(d, D, y)

    tmp = similar(y[1:D])

    @inbounds for i in 1:length(d)
        Di = D*(i-1)

        tmp .= 0.0

        for j in 1:D
            for k in 1:D
                tmp[j] += d[i][j,k] * y[Di+k]
            end
        end

        for j in 1:D
            y[Di+j] = tmp[j]
        end

        #y[(Di+1):(Di+D)] = d[i]*y[(Di+1):(Di+D)]
    end
end

# should hard-code D=2 and D=3
function backward!(ops, D, y)
    if D == 1
        backward1!(ops, y)
    else
        backwardD!(ops, D, y)
    end
end

# For D = 1
function backward1!(ops, y)

    for i in length(ops):-1:1
        op = ops[i]

        y[op.i] = op.wii*y[op.i] + op.wji*y[op.j]
    end
end

#=
# This is the idealized function, but it is not fast enough.
=#
#=
function backwardD!(ops, D, y)

    for i in length(ops):-1:1
        op = ops[i]
        di = D*(op.i-1)
        dj = D*(op.j-1)

        y[(di+1):(di+D)] = op.wii*y[(di+1):(di+D)] + op.wji'*y[(dj+1):(dj+D)]
    end

end
=#

function backwardD!(ops, D, y)

    tmp = similar(y[1:D])

    for i in length(ops):-1:1
        op = ops[i]

        di = D*(op.i-1)
        dj = D*(op.j-1)

        tmp .= 0

        for j in 1:D
            for k in 1:D
                tmp[j] += op.wii[j,k] * y[di+k] + op.wji[k,j]'*y[dj+k]
            end
        end

        for j in 1:D
            y[di+j] = tmp[j]
        end
    end
end

# function solver{Tv}(ops::Array{IJop,1}, d::Array, b::Array{Tv,1})
function solver(ops, d, b)
    y = copy(b)
    D = size(ops[1].wii,1)

    forward!(ops, D, y)

    diag!(d, D, y)

    backward!(ops, D, y)

    return y
end

#=======================

  Experimental Solvers

==========================#


"""
    cnt = count_nonzero_diags(a::SparseMatrixCSC)

Count number of nonzero diagonal elements in a.
"""
function count_nonzero_diags(a::SparseMatrixCSC)

    @assert a.m == a.n
    cnt = 0
    for j in 1:a.n
        for i in a.colptr[j]:(a.colptr[j+1]-1)
            cnt += (a.rowval[i] == j) && (!iszero(a.nzval[i]))
        end
    end
    return cnt
end


"""
    D, R = diags_and_rest(a::SparseMatrixCSC{Tv,Ti})

Create a diagonal matrix D and a sparse matrix R with zero diagonal
such that D+R = a.  Requires a to be square.
Actually returns the vector of the diagonal D.
So, if you want a matrix, use Diagonal(D)
"""
function diags_and_rest(a::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

    @assert a.m == a.n
    cnt = count_nonzero_diags(a)
    dv = zeros(Tv, a.n)

    colptr = similar(a.colptr)
    rowval = Vector{Ti}(undef, length(a.rowval) - cnt)
    nzval = Vector{Tv}(undef, length(a.rowval) - cnt)

    ptr = one(Ti)

    for j in 1:a.n
        colptr[j] = ptr
        for i in a.colptr[j]:(a.colptr[j+1]-1)
            if a.rowval[i] == j
                dv[j] = a.nzval[i]
            else
                rowval[ptr] = a.rowval[i]
                nzval[ptr] = a.nzval[i]
                ptr += 1
            end
        end
    end

    colptr[end] = ptr

    R = SparseMatrixCSC(a.m, a.n, colptr, rowval, nzval)

    return dv, R
end

#================================================

  Triangular Solves for matrices triangular up to an order.
  Will need to compare to removing diagonals
  and permutation followed by triangular Solves

  =================================================#


function backsolve!(L::SparseMatrixCSC, b::Vector)

    @assert length(b) == L.n
    @inbounds for j in L.n:-1:1
        for ii in (L.colptr[j]+1):(L.colptr[j+1]-1)
            b[j] -= L.nzval[ii] * b[L.rowval[ii]]
        end
        b[j] /= L.nzval[L.colptr[j]]
    end
end

function backsolve(L::SparseMatrixCSC, b::Vector)
    x = copy(b)
    backsolve!(L, x)
    return x
end

function backsolve(L::SparseMatrixCSC, b::Vector, p::Vector, ip::Vector)
    x = b[p]
    backsolve!(L, x)
    return x[ip]
end

function forwardsolve(L::SparseMatrixCSC, b::Vector, p::Vector, ip::Vector)
    x = b[p]
    forwardsolve!(L, x)
    return x[ip]
end

function solver(sdd_tri::SparseMatrixCSC, d_for_ldl::Vector, p::Vector, ip::Vector, b)
    y = forwardsolve(sdd_tri, b, p, ip)

    diag!(d, D, y)

    backward!(ops, D, y)

    return y
end

function forwardsolve!(L::SparseMatrixCSC, b::Vector)

    @assert length(b) == L.n
    @inbounds for j in 1:L.n

        b[j] /= L.nzval[L.colptr[j]]

        for ii in (L.colptr[j]+1):(L.colptr[j+1]-1)
            i = L.rowval[ii]
            b[i] -= b[j] * L.nzval[ii]
        end
    end

end


function disordered_sparse(L, order)
    colptr = similar(L.colptr)
    rowval = similar(L.rowval)
    nzval = similar(L.nzval)
    n = L.n
    ptr = 1
    for i in 1:n
        col = order[i]
        colptr[i] = ptr
        lead = ptr
        rowval[lead] = col
        ptr += 1
        for j in L.colptr[col]:(L.colptr[col+1]-1)
            row = L.rowval[j]
            if row != col
                rowval[ptr] = L.rowval[j]
                nzval[ptr] = L.nzval[j]
                ptr += 1
            else
                nzval[lead] = L.nzval[j]
            end
        end

    end
    colptr[n+1] = ptr
    return colptr, rowval, nzval
end

#=
Solves with the order built in to leading entries of rowval:
the semantics are not exactly what they are for sparse matrices.
rowval[colptr[i]] is the ith column in the elimination order.
nzval[colptr[i]] is the corresponding diagonal entry.
rowval[colptr[i]+1] up to the next are the other entries in that column.
=#
function forwardsolve!(colptr::Vector{Ti}, rowval::Vector{Ti},
    nzval::Vector{Tv}, b::Vector{Tv}) where {Ti,Tv}
    n = length(colptr)-1

    @assert length(b) == n
    @inbounds for j in 1:n
        col = rowval[colptr[j]]

        b[col] /= nzval[colptr[j]]

        for ii in (colptr[j]+1):(colptr[j+1]-1)
            i = rowval[ii]
            b[i] -= b[col] * nzval[ii]
        end
    end
end

function backsolve!(colptr::Vector{Ti}, rowval::Vector{Ti},
    nzval::Vector{Tv}, b::Vector{Tv}) where {Ti,Tv}
    n = length(colptr)-1

    @assert length(b) == n
    @inbounds for j in n:-1:1
        col = rowval[colptr[j]]

        for ii in (colptr[j]+1):(colptr[j+1]-1)
            b[col] -= nzval[ii] * b[rowval[ii]]
        end
        b[col] /= nzval[colptr[j]]
    end
end


"""
    x = backsolve(L, order, b)

Does a backsolve (as in upper triangular version) in a matrix that
is lower-triangular under the order b.

Not necessarily an efficient routine, but one we will try to start.
"""
function backsolve!(L::SparseMatrixCSC, order::Vector, b::Vector)

    @assert length(b) == L.n == length(order)
    @inbounds for jj in L.n:-1:1
        j = order[jj]
        for ii in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[ii]
            if j != i
                b[j] -= L.nzval[ii] * b[i]
            end
        end
        b[j] /= L[j,j]
    end
end


function backsolve(L::SparseMatrixCSC, order::Vector, b::Vector)
    x = copy(b)
    backsolve!(L, order, x)
    return x
end

"""
    x = forwardsolve(L, order, b)

Does a forwardsolve (as in upper triangular version) in a matrix that
is lower-triangular under the order b.

Not necessarily an efficient routine, but one we will try to start.
"""
function forwardsolve!(L::SparseMatrixCSC, order, b)

    @assert length(b) == L.n == length(order)
    @inbounds for jj in 1:L.n
        j = order[jj]
        b[j] /= L[j,j]

        for ii in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[ii]
            if i != j
               b[i] -= b[j] * L.nzval[ii]
            end
        end
    end
end

function forwardsolve(L::SparseMatrixCSC, order, b)
    x = copy(b)
    forwardsolve!(L, order, x)
    return x
end


"""
For separted diagonal and lower triangular part, R
"""
function forwardsolve!(D::Vector, L::SparseMatrixCSC, order, b)

    @assert length(b) == L.n == length(order) == length(D)
    @inbounds for jj in 1:L.n
        j = order[jj]
        b[j] /= D[j]

        for ii in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[ii]
            b[i] -= b[j] * L.nzval[ii]
        end
    end
end

function forwardsolve(D::Vector, L::SparseMatrixCSC, order, b)
    x = copy(b)
    forwardsolve!(D, L, order, x)
    return x
end

function backsolve!(D::Vector, L::SparseMatrixCSC, order, b)

    @assert length(b) == L.n == length(order) == length(D)
    @inbounds for jj in L.n:-1:1
        j = order[jj]
        for ii in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[ii]
            b[j] -= L.nzval[ii] * b[i]
        end
        b[j] /= D[j]
    end
end


function backsolve(D::Vector, L::SparseMatrixCSC, order, b)
    x = copy(b)
    backsolve!(D, L, order, x)
    return x
end
