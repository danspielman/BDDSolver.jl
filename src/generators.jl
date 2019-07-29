function randOrth(d)
    q = qr(randn(d,d)).Q
    q = q * Diagonal(rand([1;-1],d))
    return q
end

"""
    Q = randOrth(d, pert)

Produces a random orthogonal matrix close to I.
The smaller pert is, the closer to I.
The diagonals are always positive.
"""
function randOrth(d, pert)
    q = qr(eye(d)+pert*randn(d,d)).Q
    q = q * Diagonal(sign.(diag(q)))
    return q
end

function randFullBDD(n,d)
    A = triu([randOrth(d) for i in 1:n, j in 1:n],1)
    for i in 1:n
        A[i,i] = eye(d)*(0.001 + (n-1)/2)
    end
    B = BlockCSC(A)
    B = B + transpose(B)
end

function randOrthWt(a,d)
    #deg = sum(a,dims=1)
    n = size(a,1)
    ai,aj,av = findnz(triu(a))
    vq = [randOrth(d)*av[i] for i in 1:length(av)]

    ai = vcat(ai,[n])
    aj = vcat(aj,[n])
    vq = vcat(vq,[zeros(d,d)])

    B = BlockCSC(sparse([ai;aj],[aj;ai],[vq;collect.(transpose.(vq))],n,n))
    #B = B + transpose(B)

    E = one(typeof(B[1,1]))

    sp = BlockCSC(sparse(1:n,1:n,[E*sum(opnorm.(a[:,i])) for i in 1:n]))

    return B + sp
end

"""
    B = randOrthWtPotential(a,d, pert)

Produces a bdd matrix with orthogonal matrices, slightly perturbed from a potential.
If the potential is zero, then the resulting matrix is singular.
Note that if there are very few edges, then the matrix can be singular anyway.
This motivates randBDDWtPotential, which is never singular.
"""
function randOrthWtPotential(a,d, pert)
    #deg = sum(a,dims=1)
    n = size(a,1)
    ai,aj,av = findnz(triu(a))

    vr = [randOrth(d) for i in 1:n]

    vq = [-vr[ai[i]]*vr[aj[i]]'*randOrth(d,pert)*av[i] for i in 1:length(av)]

    ai = vcat(ai,[n])
    aj = vcat(aj,[n])
    vq = vcat(vq,[zeros(d,d)])

    B = BlockCSC(sparse(ai,aj,vq,n,n))
    B = B + transpose(B)

    dg = [eye(d)*sum(opnorm.(B[:,i])) for i in 1:n]
    B = B + BlockCSC(sparse(Diagonal(dg)))
end

function randBDDWtPotential(a,d, pert)
    #deg = sum(a,dims=1)
    n = size(a,1)
    ai,aj,av = findnz(triu(a))

    vr = [randOrth(d) for i in 1:n]

    vq = [(-vr[ai[i]]*vr[aj[i]]' + randn(d,d)*pert)*av[i] for i in 1:length(av)]

    ai = vcat(ai,[n])
    aj = vcat(aj,[n])
    vq = vcat(vq,[zeros(d,d)])

    B = BlockCSC(sparse(ai,aj,vq,n,n))
    B = B + transpose(B)

    dg = [eye(d)*sum(opnorm.(B[:,i])) for i in 1:n]
    B = B + BlockCSC(sparse(Diagonal(dg)))
end

function randBDDWt(a,d)
    n = size(a,1)
    ai,aj,av = findnz(triu(a))
    vq = [randn(d,d) for i in 1:length(av)]
    ai = vcat(ai,[n])
    aj = vcat(aj,[n])
    vq = vcat(vq,[zeros(d,d)])
    B = BlockCSC(sparse(ai,aj,vq,n,n))
    B = B + transpose(B)

    dg = [eye(d)*sum(opnorm.(B[:,i])) for i in 1:n]
    B = B + BlockCSC(sparse(Diagonal(dg)))
end
