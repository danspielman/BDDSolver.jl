import Laplacians.condNumber
import Laplacians.sddmWrapLap

using StaticArrays

#=

A variations of the approxChol Laplacian solver
by Rasmus Kyng and Daniel A. Spielman.
It solves systems in non-singular BDD (Block Diagonally Dominant) Matrices.
This algorithm is an implementation of an approximate edge-by-edge elimination
algorithm inspired by the Approximate Gaussian Elimination algorithm of
Kyng and Sachdeva.


TODO:

- get rid of mat to analyze, and just build it in to llmatp
- Check on op norm, where faster and how call fastest.

 - Figure out what to do with singular Matrices
 - Make work with full matrices.

 - Test on approximate potential orths
 - Remove logging

=#



"""
    params = AC_Params(queue_forward, stag_test)

queue_forward=true if we use the forward queue (the original)
stag_test is a parameter for PCG, default 5.
"""
mutable struct AC_Params
    queue_forward::Bool
    stag_test::Integer
    useop::Bool
    perm::Bool
end

AC_Params() = AC_Params(true, 5, true, false)
AC_Params(b::Bool) = AC_Params(b, 5, true, false)
AC_Params(qf::Bool, uop::Bool) = AC_Params(qf, 5, uop, false)


#=
  For small matrices, the following is faster than "opnorm".
=#
_ournorm(x::StaticArrays.SArray) = sqrt(maximum(eigvals(x*x')))
ournorm(x::Real) = abs(x)
ournorm(mat::SMatrix{1,1}) = abs(mat[1])
ournorm(mat::SMatrix{2,2}) = _ournorm(mat)
ournorm(mat::SMatrix{3,3}) = _ournorm(mat)
ournorm(mat::SMatrix{4,4}) = _ournorm(mat)
ournorm(mat::SMatrix{5,5}) = _ournorm(mat)
ournorm(mat::SMatrix{6,6}) = _ournorm(mat)
ournorm(mat) = opnorm(mat)



function LLmata(a::SparseMatrixCSC{Tval,Tind}) where {Tind,Tval}
    n = size(a,1)
    m = nnz(a)

    degs = zeros(Tind,n)
    ptrs = copy(a.colptr)

    cols = a.colptr[1:n]
    llelems = Vector{LLa{Tind,Tval}}(undef, m)

    for i in 1:n
        degs[i] = a.colptr[i+1] - a.colptr[i]

        for ind in a.colptr[i]:(a.colptr[i+1]-1)
            j = a.rowval[ind]

            if j < i
                v = a.nzval[ind]

                if ptrs[i] < a.colptr[i+1]-1
                    llelems[ptrs[i]] = LLa(j, v, ptrs[i]+1, ptrs[j])
                else
                    llelems[ptrs[i]] = LLa(j, v, zero(Tind), ptrs[j])
                end

                if ptrs[j] < a.colptr[j+1]-1
                    llelems[ptrs[j]] = LLa(i, v, ptrs[j]+1, ptrs[i])
                else
                    llelems[ptrs[j]] = LLa(i, v, zero(Tind), ptrs[i])
                end

                ptrs[j] += 1
                ptrs[i] += 1
            end
        end

    end

    return LLmata{Tind,Tval}(n, degs, cols, llelems)
end

function print_ll_col(llmat::LLmata, i::Int)
    next = llmat.cols[i]
    while next != 0
      ll = llmat.lles[next]
      println("col $i, row $(ll.row) : $(ll.val)")

      next = ll.next
    end
end





#=============================================================

The approximate factorization

=============================================================#

function get_ll_col(llmat::LLmata{Tind,Tval},
    i,
    colspace::Vector{LLcol{Tind,Tval}}) where {Tind,Tval}
  
      ptr = llmat.cols[i]
      len = 0
      while ptr != 0
  
          if llmat.lles[ptr].val > zero(Tval)
              len = len+1
  
              item = LLcol(llmat.lles[ptr].row, ptr, llmat.lles[ptr].val)
              if (len > length(colspace))
                  push!(colspace,item)
              else
                  colspace[len] = item
              end
          end
  
          ptr = llmat.lles[ptr].next
      end
  
      return len
  end


#=
Add together entries of the column that are in the same row.
Decrease the degrees of vertices when their entries are added.
Returns the number of nonzero entries in the column after the compression, `ptr`.
=#



function compress_col!(llmat::LLmata{Tind,Tval},
    colspace::Vector{LLcol{Tind,Tval}},
    len::Tind,
    pq::acPQ{Tind},
    col::Int,
    compressions) where {Tind,Tval}
  
    o1 = Base.Order.ord(isless, x->x.row, false, Base.Order.Forward)

    sort!(colspace, one(len), len, QuickSort, o1)

    c = colspace

    ptr = 0
    currow = c[1].row
    curval = c[1].val
    curptr = c[1].ptr

    for i in 2:len

        if c[i].row != currow

            ptr = ptr+1
            c[ptr] = LLcol(currow, curptr, curval)  

            currow = c[i].row
            curval = c[i].val
            curptr = c[i].ptr

        else

            curval = curval + c[i].val
            push!(compressions, Compression(col ,curptr, c[i].ptr))

            zeroval(llmat, llmat.lles[c[i].ptr].reverse)  
            pqDec!(pq, currow)
        end

    end

    # emit the last row

    ptr = ptr+1
    c[ptr] = LLcol(currow, curptr, curval)

    o2 = Base.Order.ord(isless, x->x.val, false, Base.Order.Forward)
    sort!(colspace, one(ptr), ptr, QuickSort, o2)

    return ptr
end



const order_forward = Base.Order.ord(isless, identity, false, Base.Order.Forward)

"""
    a = mat_to_analyze(B::BlockCSC)

Produces an adjacency matrix of a graph whose edge weights are the norms
of matrices in B.  This could probably be accelerated.  
"""
function mat_to_analyze(B::BlockCSC)

    mat = B.mat

    a = SparseMatrixCSC(mat.m, mat.n, copy(mat.colptr), copy(mat.rowval), ournorm.(mat.nzval))
    for i in 1:a.n
        a[i,i] = 0.0
    end
    dropzeros!(a)

    return a
end




function zeroval(llmat::LLmata{Tind,Tval}, revj::Tind) where {Tind, Tval}
    llmat.lles[revj] = LLa(llmat.lles[revj].row, 
            zero(Tval), 
            llmat.lles[revj].next ,
            llmat.lles[revj].reverse )
end

#=

Create edge (k,j), using the memory for (i,j)
=#
function rotate_edges(llmat::LLmata,k,j,ptr,revj,newEdgeVal)
    llmat.lles[revj] = LLa(k, newEdgeVal, llmat.lles[revj].next, ptr)


    # fix row j in col k
    khead = llmat.cols[k]
    llmat.cols[k] = ptr

    llmat.lles[ptr] = LLa(j, newEdgeVal, khead, revj)

end


function compute_elim(a::LLmata{Tind,Tval}; params = AC_Params()) where {Tind,Tval}

    #logstart(fn)

    n = a.n

    ijks = Vector{IJKop2{Tind,Tval}}(undef,0)

    #if params.queue_forward
        pq = acPQ(a.degs)
    #else
    #    pq = acPQ_rev(a.degs)
    #end

    colspace = Vector{LLcol{Tind,Tval}}(undef,n)
    cumspace = Vector{Tval}(undef,n)

    # elim_rank[i] is the iter in which vertex i is eliminated
    elim_rank = Vector{Tind}(undef,n) 
    order_tmp = Vector{Tind}(undef, n)
    
    it = 1

    # DAS: might be computable from other terms, in which case can remove.
    diag_tmp = Vector{Tval}(undef,n) 
    diag_adjusts = zeros(Tval, n)

    compressions = Vector{Compression}()

#    @inbounds while it < n
    while it < n

        i = pqPop!(pq)
        elim_rank[i] = it

        it = it + 1

        elim_vertex(a, i, ijks, colspace, cumspace,  
        diag_tmp, diag_adjusts, order_tmp, pq, compressions, params)

    end

    last_col = pqPop!(pq)
    elim_rank[last_col] = n

    return ijks, diag_adjusts, elim_rank, last_col, compressions
end



function elim_vertex(llmat::LLmata{Tind,Tval}, 
    i::Int, 
    ijks::Vector{IJKop2{Tind,Tval}},
    colspace::Vector{LLcol{Tind,Tval}}, 
    cumspace::Vector{Tval}, 
    diag_tmp::Vector{Tval}, 
    diag_adjusts::Vector{Tval},
    order_tmp::Vector{Tind},
    pq, 
    compressions::Vector{Compression},
    params::AC_Params) where {Tind,Tval}


    len = get_ll_col(llmat, i, colspace)
    if len == 0
        return
    end

    len = compress_col!(llmat, colspace, len, pq, i, compressions)  
    if len == 0
        return
    end

    #println("$(i): $(len)")

    # csum is the weighted degree of a vertex.
    # cumspace holds the cumulative sums of weights of edges, so that we can sample
    csum = zero(Tval)
    for ii in 1:len
        csum = csum + colspace[ii].val
        cumspace[ii] = csum
    end

    # we store adjustments to diagonals locally in diag_tmp, and then write them to 
    # diagonal adjusts later.  This line sets the default adjusts, for when there are no incoming
    # edges to the vertex.
    for hoff in 2:len
        diag_tmp[hoff] = -(cumspace[hoff-1]) * colspace[hoff].val / csum
    end 
      
    # Here we choose all the edges to be sampled.  
    # This could be done in bulk, maybe faster, or in line with the next loop.
    for joffset in 1:(len-1)
        r = rand() * (csum - cumspace[joffset]) + cumspace[joffset]
        order_tmp[joffset] = searchsortedfirst(cumspace,r,joffset+1,len,order_forward) 
    end

    # loop over the edges attached to vertex i, doing eliminations.
    for joffset in 1:(len-1)

        # are eliminating edge (i,j), and creating edge (j,k)

        ll = colspace[joffset]

        j = ll.row
        revj = llmat.lles[ll.ptr].reverse

        # (j,k) is the edge added.
        koff = order_tmp[joffset]
        k = colspace[koff].row

        pqInc!(pq, k)

        # this is the factor by which we multiply row i to elim row j
        op = - ll.val / csum

        # put in the new edge (j, k)
        oldEdgeVal = colspace[koff].val

        # this is what we really need to know for the factorization:
        # the probability that we chose this edge (j, k)
        probadd = oldEdgeVal / (csum-cumspace[joffset])

        # the following, commented out, form is equivalent
        # newEdgeVal = op * oldEdgeVal  / probadd  
        newEdgeVal = (csum-cumspace[joffset]) * ll.val / csum

        # Compensate diagonal k for having sampled this edge
        diag_tmp[koff] += newEdgeVal

        rotate_edges(llmat,k,j,ll.ptr,revj,newEdgeVal)

        push!(ijks, IJKop2(i,j,k, probadd, ll.ptr, colspace[koff].ptr, revj))

    end # for joffset

    # adjust diagonals from sampling
    # takes a lot of time: all cache misses
    for hoff in 2:len
        h = colspace[hoff].row
        diag_adjusts[h] += diag_tmp[hoff]
    end


    # For the degre 1 case - last edge - just adjust diagonal

    ll = colspace[len]
    j = ll.row
    revj = llmat.lles[ll.ptr].reverse

    #= should firgure out why wanted it < n
    if it < n
        pqDec!(pq, j)
    end
    =#

    pqDec!(pq, j)

    # zero out that edge for the other side.
    zeroval(llmat, revj)

    # zero in k position indicates no edge
    push!(ijks, IJKop2(i,j,0, 1.0, ll.ptr, revj, 0))

end



function bdd_setup(B::BlockCSC{D,Tv,L}, elim_rank) where {D, Tv, L}
    mat = B.mat
    
    n = size(mat,1)
    cnt = count_nonzero_diags(mat)
    
    m = nnz(mat) - cnt
    
    matstore = Vector{StaticArrays.SArray{Tuple{D,D},Tv,2,L}}(undef, m)
    matstore = zeros(StaticArrays.SArray{Tuple{D,D},Tv,2,L}, m)
    dgs = Vector{StaticArrays.SArray{Tuple{D,D},Tv,2,L}}(undef, n)
    
    diag_free_ind = 0
    
    for i in 1:n

        for ind in mat.colptr[i]:(mat.colptr[i+1]-1)
            j = mat.rowval[ind]
            
            if i == j
                dgs[i] = mat.nzval[ind]
            else
                diag_free_ind += 1

                # SOMETHING IS WRONG IF NEED TO COMMENT THIS OUT!
                #if elim_rank[i] < elim_rank[j]
                    matstore[diag_free_ind] = mat.nzval[ind]
                #end
            end
        end
    end
    
    return matstore, dgs
end



"""
This makes a BlockCSC with structural zeros and the elements of bdd that we will use
in the elimination.  It allows us to do the factorizations just once.
"""
function prepare_triangular_factor(bdd::BlockCSC{D,Tv,L}, ijks, elim_rank) where {D,Tv,L}
    n = size(bdd,1)
    m = length(ijks)

    ti = Vector{Int}(undef, m+n)
    tj = Vector{Int}(undef, m+n)
    tv = zeros(typeof(bdd[1,1]), m+n)
    ptr = 1
    i = 0
    for it in 1:m
        ijk = ijks[it]
        if ijk.i != i
            i = ijk.i
            ti[ptr] = i
            tj[ptr] = i
            #tv[ptr] = la[i,i]
            ptr += 1
        end
        ti[ptr] = ijk.i
        tj[ptr] = ijk.j
        ptr += 1
    end
    lst = argmax(elim_rank)
    ti[ptr] = lst
    tj[ptr] = lst
    bdd_tri = BlockCSC(sparse(tj,ti,tv,n,n))

    aj, ai, av = findnz(bdd.mat)
    for it in 1:length(ai)
        if elim_rank[ai[it]] <= elim_rank[aj[it]]
            bdd_tri[aj[it], ai[it]] = av[it]
        #else
        #    bdd_tri[ai[it], aj[it]] = av[it]
        end
    end

    return bdd_tri
end


function factor_bdd_ijks_plt(ijks, diag_adjusts, 
    elim_rank, 
    matstore::Vector{StaticArrays.SArray{Tuple{D,D},Tv,2,L}}, dgs, comps) where {D,Tv,L}


    Tblock = StaticArrays.SArray{Tuple{D,D},Tv,2,L}

    n = length(dgs)
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, length(ijks) + n)
    nzval = Vector{Tblock}(undef, length(ijks) + n)
    colptr_ind = 1
    rowval_ind = 1


    E = one(Tblock)

    diaginv = E

    # true when are first entry in col.  Would be better to have a cscish ijks, and skip this.
    flag = true

    comps_ind = 1

    for it in 1:length(ijks)
        ijk = ijks[it]
        
        if flag

            while (comps_ind <= length(comps)) && (comps[comps_ind].col == ijk.i) 
                matstore[comps[comps_ind].into] += matstore[comps[comps_ind].from]
                comps_ind += 1
            end

            dgs[ijk.i] += E*diag_adjusts[ijk.i]

            # place the diagonal entry
            colptr[colptr_ind] = rowval_ind
            colptr_ind += 1
            rowval[rowval_ind] = ijk.i
            nzval[rowval_ind] = dgs[ijk.i]
            rowval_ind += 1

            diaginv = inv(dgs[ijk.i])   

            flag = false
        end

        op = - matstore[ijk.ptr_ji] * diaginv     

        dgs[ijk.j] += op * (matstore[ijk.ptr_ji])'

       # push out an off-diagonal entry
       rowval[rowval_ind] = ijk.j

       nzval[rowval_ind] = matstore[ijk.ptr_ji]
       rowval_ind += 1

        if ijk.k != 0 # if not a degree 1 elim: again better to use cscish ijks.

            if elim_rank[ijk.j] < elim_rank[ijk.k]
                matstore[ijk.ptr_kj] = (op * ( matstore[ijk.ptr_ki])')' / ijk.prob
            else
                matstore[ijk.ptr_ji] = op * ( matstore[ijk.ptr_ki])'/ ijk.prob
            end
            
        else

            flag = true
        end
    end

    lst = argmax(elim_rank)

    colptr[colptr_ind] = rowval_ind
    rowval[rowval_ind] = lst

    nzval[rowval_ind] = dgs[lst] + E*diag_adjusts[lst]
 
    colptr[colptr_ind+1] = rowval_ind+1

    return PermLTCSC(colptr, rowval, nzval)

end

# this will be removed from the main code line
function AC_BDD_precon(bdd; verbose=false, params = AC_Params())
  
    t1 = time()
    t0 = t1

    a = mat_to_analyze(bdd)
    if verbose
        println("Time to construct mat to analyze ", time()-t1)
        t1 = time()
    end
    
    ijks, diag_adjusts, elim_rank, last_col, comps = compute_elim(LLmata(a))
    if verbose
        println("Time to compute elimination structure ", time()-t1)
        t1 = time()
    end


    matstore, dgs = bdd_setup(bdd, elim_rank)
    if verbose
        println("Time to prepare for elimination ", time()-t1)
        t1 = time()
    end

    ltcsc = factor_bdd_ijks_plt(ijks, diag_adjusts, elim_rank, matstore, dgs, comps)
    if verbose
        println("Time to eliminate ", time()-t1)
        t1 = time()
    end

    F(b) = solver(ltcsc, b)

    if verbose
        println("Total solver build time: ", time()-t0)
        println("Ratio of operator edges to original edges: ", 2 * length(ltcsc.nzval) / nnz(bdd))
    end

    return F
end




function solver(B, ltcsc; 
    pcg_params = PCG_params())

    F!(b) = solver!(ltcsc, b)

    #=
    f1(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, 
        verbose=verbose_, 
        pcgIts=pcgIts_) = 
    pcg(Bp, b[elim_order], F, tol=tol, maxits=maxits, maxtime=maxtime, 
        pcgIts=pcgIts, 
        verbose=verbose, 
        stag_test = params.stag_test)[elim_rank]
=#

    f1(b;tol=pcg_params.tol,
        maxits=pcg_params.maxits, 
        maxtime=pcg_params.maxtime, 
        verbose=pcg_params.verbose, 
        pcgIts=pcg_params.pcgIts) = 
    pcg_inplace(B, b, F!, 
        tol=tol, 
        maxits=maxits, 
        maxtime=maxtime, 
        pcgIts=pcgIts, 
        verbose=verbose, 
        stag_test = pcg_params.stag_test)
        

    return f1
end

function permed_solver(B, ltcsc, elim_rank::Vector{Int}; 
    pcg_params = PCG_params())

    elim_order = invperm(elim_rank)
    Bp = BlockCSC(B.mat[elim_order, elim_order])

    # note: the following does not copy, but directs pointer
    ltcsc_p = LTCSC(ltcsc.colptr, elim_rank[ltcsc.rowval], ltcsc.nzval)

    F(b) = solver(ltcsc_p, b)

    #=
    f1(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, 
        verbose=verbose_, 
        pcgIts=pcgIts_) = 
    pcg(Bp, b[elim_order], F, tol=tol, maxits=maxits, maxtime=maxtime, 
        pcgIts=pcgIts, 
        verbose=verbose, 
        stag_test = params.stag_test)[elim_rank]
=#

    f1(b; tol=pcg_params.tol,
        maxits=pcg_params.maxits, 
        maxtime=pcg_params.maxtime, 
        verbose=pcg_params.verbose, 
        pcgIts=pcg_params.pcgIts) = 
    pcg_perm(elim_rank, Bp, copy(b), F, 
        PCG_params(verbose, tol, maxits, maxtime, pcgIts, pcg_params.stag_test )
        )

    return f1
end


function pcg_perm(perm, 
    mat::BlockCSC{D,Tv,L}, 
    b::Vector{SArray{Tuple{D},Tv,1,D}}, 
    pre::Function,
    pcg_params::PCG_params)  where {D,Tv,L}

    bp = b[perm]
    x = similar(b)

    x[perm] = pcg(mat, bp, pre, tol=pcg_params.tol, 
        maxits=pcg_params.maxits, 
        maxtime=pcg_params.maxtime, 
        pcgIts=pcg_params.pcgIts, 
        verbose=pcg_params.verbose, 
        stag_test = pcg_params.stag_test)
        
    return x

end

"""
    solver = approxChoBDD(bdd); x = solver(b);
    solver = approxCholBDD(bdd; tol::Real=1e-6, maxits=1000, maxtime=Inf, verbose=false, pcgIts=Int[])

A heuristic by Daniel Spielman inspired by the linear system solver in https://arxiv.org/abs/1605.02353 by Rasmus Kyng and Sushant Sachdeva.  Whereas that paper eliminates vertices one at a time, this eliminates edges one at a time.  It is probably possible to analyze it.
It solves systems of equations in nonsingular Block Diagonally Dominant Matrices.
ADD NOTES ON HOW TO CONSTRUCT THESE.

For more info, see http://danspielman.github.io/Laplacians.jl/latest/usingSolvers/index.html
"""
function approxCholBDD(bdd;
    tol=1e-6,
    maxits=1000,
    maxtime=Inf,
    verbose=false,
    pcgIts=Int[],
    params = AC_Params())
  
    pcg_params = PCG_params(verbose, 
        tol, maxits, maxtime, pcgIts, 
        params.stag_test)

    tol_ =tol
    maxits_ =maxits
    maxtime_ =maxtime
    verbose_ =verbose
    pcgIts_ =pcgIts

    t1 = time()
    t0 = t1

    #M = sparse(bdd)

    a = mat_to_analyze(bdd)
    if verbose
        println("Time to construct mat to analyze ", time()-t1)
        t1 = time()
    end
    
    ijks, diag_adjusts, elim_rank, last_col, comps = compute_elim(LLmata(a))
    if verbose
        println("Time to compute elimination structure ", time()-t1)
        t1 = time()
    end


    matstore, dgs = bdd_setup(bdd, elim_rank)
    if verbose
        println("Time to prepare for elimination ", time()-t1)
        t1 = time()
    end

    ltcsc = factor_bdd_ijks_plt(ijks, diag_adjusts, elim_rank, matstore, dgs, comps)
    if verbose
        println("Time to eliminate ", time()-t1)
        t1 = time()
    end

    permute_outside = params.perm

    if permute_outside

        f1 = permed_solver(bdd, ltcsc, elim_rank,
            pcg_params = pcg_params)



        if verbose
            println("Time to permute ", time()-t1)
            t1 = time()
        end
    
    else
        
        f1 = solver(bdd,  ltcsc,
            pcg_params = pcg_params)

    end   

    if verbose
        println("Total solver build time: ", time()-t0)
        println("Ratio of operator edges to original edges: ", 2 * length(ltcsc.nzval) / nnz(bdd))
    end

    return f1
end



