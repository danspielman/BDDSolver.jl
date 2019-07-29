#=
  Structs for the approxChol solver
=#


"""
  LLp elements are all in the same column.
  row tells us the row, and val is the entry.
  val is set to zero for some edges that we should remove.
  next gives the next in the column.  It points to itself to terminate.
  reverse is the index into lles of the other copy of this edge,
  since every edge is stored twice as we do not know the order of elimination in advance.
"""
mutable struct LLp{Tind,Tval}
    row::Tind
    val::Tval
    next::LLp{Tind,Tval}
    reverse::LLp{Tind,Tval}

    LLp{Tind,Tval}() where {Tind,Tval} = (x = new(zero(Tind), zero(Tval)); x.next = x; x.reverse = x)
    LLp{Tind,Tval}(row, val, next, rev) where {Tind,Tval} = new(row, val, next, rev)
    LLp{Tind,Tval}(row, val) where {Tind,Tval} = (x = new(row, val); x.next = x; x.reverse = x)
    LLp{Tind,Tval}(row, val, next) where {Tind,Tval} = (x = new(row, val, next); x.reverse = x)
end

"""
  LLmatp is the data structure used to maintain the matrix during elimination.
  It stores the elements in each column in a singly linked list (only next ptrs)
  Each element is an LLp (linked list pointer).
  The head of each column is pointed to by cols.

  We probably can get rid of degs - as it is only used to store initial degrees.
"""
struct LLmatp{Tind,Tval}
    n::Int64
    degs::Array{Tind,1}
    cols::Array{LLp{Tind,Tval},1}
    lles::Array{LLp{Tind,Tval},1}
end

# The followin are for an array-based structure
struct LLa{Tind,Tval}
    row::Tind
    val::Tval
    next::Tind
    reverse::Tind
end

struct LLmata{Tind,Tval}
    n::Int64
    degs::Vector{Tind}
    cols::Vector{Tind}
    lles::Vector{LLa{Tind,Tval}}
end

struct LLcol{Tind,Tval}
    row::Tind
    ptr::Tind
    val::Tval
end


# the following code might not work
bdd_type(B::BlockCSC{D,Tval} where {D, Tval}) = (D, Tval)
bdd_type(B::SparseMatrixCSC{Tval,Tind} where {Tval,Tind}) = (1, Tval)



#=============================================================

LDLinv

=============================================================#


struct IJop{Ti,Tv}
    i::Ti
    j::Ti
    wii::Tv
    wji::Tv
end

"""
Records that col i was used to elim entry in row j,
and that it was swapped into an entry in j, k.
"""
struct IJKop{Ti,Tv}
    i::Ti
    j::Ti
    k::Ti
    prob::Tv
end

"""
  LDLinv contains the information needed to solve the Laplacian systems.
  It does it by applying Linv, then Dinv, then Linv (transpose).
  But, it is specially constructed for this particular solver.
  It does not explicitly make the matrix triangular.
  Rather, col[i] is the name of the ith col to be eliminated
"""
mutable struct LDLinv{Tind,Tval}
    col::Array{Tind,1}
    colptr::Array{Tind,1}
    rowval::Array{Tind,1}
    fval::Array{Tval,1}
    d::Array{Tval,1}
end

abstract type AbsLTCSC{Tind, Tval} end

"""
A data type for holding a permutation of a lower-triangular matrix,
in a CSC-ish format.  The fields are colptr`, `rowval`, `nzval`.
The entries of the ith columns in the order are stored in rowval
and nzval in entries `colptr[i]`` through `colptr[i+1]-1`.  The identity
of the column is in `roval[colptr[i]]`, and its diagonal entry is in nzval.
No guarantees are made about the orders of other entries in that column.
"""
struct PermLTCSC{Tind,Tval} <: AbsLTCSC{Tind, Tval}
    colptr::Vector{Tind}
    rowval::Vector{Tind}
    nzval::Vector{Tval}
end

# similar type, but lower triangular although not nec sorted
struct LTCSC{Tind,Tval} <: AbsLTCSC{Tind, Tval}
    colptr::Vector{Tind}
    rowval::Vector{Tind}
    nzval::Vector{Tval}
end

#=============================================================

ApproxCholPQ
the data strcture we use to keep track of degrees

=============================================================#

struct acPQ_elem{Tind}
    prev::Tind
    next::Tind
    key::Tind
end

"""
  An approximate priority queue.
  Items are bundled together into doubly-linked lists with all approximately the same key.
  minlist is the min list we know to be non-empty.
  It should always be a lower bound.
  keyMap maps keys to lists
"""
mutable struct acPQ{Tind}
    elems::Array{acPQ_elem{Tind},1} # indexed by node name
    lists::Array{Tind,1}
    minlist::Int
    nitems::Int
    n::Int
end

mutable struct acPQ_rev{Tind}
    elems::Array{acPQ_elem{Tind},1} # indexed by node name
    heads::Array{Tind,1}  # pop from heads: prev points to head
    tails::Array{Tind,1}  # add to tails, next points to tail
    minlist::Int
    nitems::Int
    n::Int
end

mutable struct randishPQ{Tind}
    elems::Array{acPQ_elem{Tind},1} # indexed by node name
    heads::Array{Tind,1}  # pop from heads: prev points to head
    tails::Array{Tind,1}  # add to tails, next points to tail
    minlist::Int
    nitems::Int
    n::Int
end

struct NodeDeg{Tind}
    node::Tind
    deg::Tind
end

mutable struct RandPQ{Tind}
    elems::Vector{NodeDeg{Tind}}
    starts::Vector{Tind}
    nodeptr::Vector{Tind}
    cur::Tind
end



struct Compression
    col::Int
    into::Int
    from::Int
end

struct IJKop2{Ti,Tv}
    i::Ti
    j::Ti
    k::Ti
    prob::Tv
    ptr_ji::Ti
    ptr_ki::Ti
    ptr_kj::Ti
    #ptr_jk::Ti equals ptr_ji
end

struct PCG_params
    verbose::Bool
    tol::Float64
    maxits::Float64
    maxtime::Float64
    pcgIts::Vector{Int}
    stag_test::Integer
end

PCG_params() = PCG_params(false, 1e-6, Inf, Inf, Int[], 5)
