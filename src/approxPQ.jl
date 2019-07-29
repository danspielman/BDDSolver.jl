
#=============================================================

ApproxCholPQ
It only implements pop, increment key, and decrement key.
All nodes with degrees 1 through n appear in their own doubly-linked lists.
Nodes of higher degrees are bundled together.

Also have nodes of degree 0, and lists now has a 1+ everywhere.

Will need to rename, as it is different from the version in Laplacians.
Maybe add a 0 to show that is allowed.

=============================================================#


function keyMap(x, n)
    return x <= n ? x : n + div(x,n)
end

function acPQ(a::Vector{Tind}) where {Tind}

    n = length(a)
    elems = Array{acPQ_elem{Tind}}(undef,n)
    lists = zeros(Tind, 2*n+2)
    minlist = one(n)

    for i in 1:length(a)
        key = a[i]
        head = lists[1+keyMap(key, n)]

        if head > zero(Tind)
            elems[i] = acPQ_elem{Tind}(zero(Tind), head, key)

            elems[head] = acPQ_elem{Tind}(i, elems[head].next, elems[head].key)
        else
            elems[i] = acPQ_elem{Tind}(zero(Tind), zero(Tind), key)

        end

        lists[1+keyMap(key, n)] = i
    end

    return acPQ(elems, lists, minlist, n, n)
end

function pqPop!(pq::acPQ{Tind}) where {Tind}
    if pq.nitems == 0
        error("ApproxPQ is empty")
    end
    while pq.lists[1+pq.minlist] == 0
        pq.minlist = pq.minlist + 1
    end
    i = pq.lists[1+pq.minlist]
    next = pq.elems[i].next


    pq.lists[1+pq.minlist] = next
    if next > 0
        pq.elems[next] = acPQ_elem(zero(Tind), pq.elems[next].next, pq.elems[next].key)
    end

    pq.nitems -= 1

    return i
end

function pqMove!(pq::acPQ{Tind}, i, newkey, oldlist, newlist) where {Tind}

    prev = pq.elems[i].prev
    next = pq.elems[i].next

    # remove i from its old list
    if next > zero(Tind)
        pq.elems[next] = acPQ_elem{Tind}(prev, pq.elems[next].next, pq.elems[next].key)
    end
    if prev > zero(Tind)
        pq.elems[prev] = acPQ_elem{Tind}(pq.elems[prev].prev, next, pq.elems[prev].key)

    else
        pq.lists[1+oldlist] = next
    end

    # insert i into its new list
    head = pq.lists[1+newlist]
    if head > 0
        pq.elems[head] = acPQ_elem{Tind}(i, pq.elems[head].next, pq.elems[head].key)
    end
    pq.lists[1+newlist] = i

    pq.elems[i] = acPQ_elem{Tind}(zero(Tind), head, newkey)

    #return Void
end

"""
    Decrement the key of element i
    This could crash if i exceeds the maxkey
"""
function pqDec!(pq::acPQ{Tind}, i) where {Tind}

    oldlist = keyMap(pq.elems[i].key, pq.n)
    newlist = keyMap(pq.elems[i].key - one(Tind), pq.n)

    if newlist != oldlist

        pqMove!(pq, i, pq.elems[i].key - one(Tind), oldlist, newlist)

        if newlist < pq.minlist
            pq.minlist = newlist
        end

    else
        pq.elems[i] = acPQ_elem{Tind}(pq.elems[i].prev, pq.elems[i].next, pq.elems[i].key - one(Tind))
    end

    #return Void
end

"""
    Increment the key of element i
    This could crash if i exceeds the maxkey
"""
function pqInc!(pq::acPQ{Tind}, i) where {Tind}

    oldlist = keyMap(pq.elems[i].key, pq.n)
    newlist = keyMap(pq.elems[i].key + one(Tind), pq.n)

    if newlist != oldlist

        pqMove!(pq, i, pq.elems[i].key + one(Tind), oldlist, newlist)

    else
        pq.elems[i] = acPQ_elem{Tind}(pq.elems[i].prev, pq.elems[i].next, pq.elems[i].key + one(Tind))
    end

    #return Void
end


function acPQ_rev(a::Vector{Tind}) where {Tind}

    #logwrite(fn,"pq init: $(a)")

    n = length(a)
    elems = Array{acPQ_elem{Tind}}(undef, n)
    heads = zeros(Tind, 2*n+2)
    tails = zeros(Tind, 2*n+2)
    minlist = one(n)

    for i in 1:length(a)
        key = a[i]
        tail = tails[1+key]

        if tail == zero(Tind) # if list is empty
            elems[i] = acPQ_elem{Tind}(zero(Tind), zero(Tind), key)
            tails[1+key] = i
            heads[1+key] = i
        else

            elems[i] = acPQ_elem{Tind}(tail, zero(Tind), key)

            # the following just modifies elems[tail] to point to i
            elems[tail] = acPQ_elem{Tind}(elems[tail].prev, i, elems[tail].key)

            tails[1+key] = i
        end

    end

    return acPQ_rev(elems, heads, tails, minlist, n, n)
end

function pqPop!(pq::acPQ_rev{Tind}) where {Tind}

    if pq.nitems == 0
        error("ApproxPQ is empty")
    end
    while pq.heads[1+pq.minlist] == 0
        pq.minlist = pq.minlist + 1
    end
    i = pq.heads[1+pq.minlist]
    next = pq.elems[i].next


    pq.heads[1+pq.minlist] = next
    if next > 0
        # just changing the first field
        pq.elems[next] = acPQ_elem(zero(Tind), pq.elems[next].next, pq.elems[next].key)
    else # removed the last element
        pq.tails[1+pq.minlist] = 0
    end

    pq.nitems -= 1

    #logwrite(fn,"pq pop: $(i)")

    return i
end

function pqMove!(pq::acPQ_rev{Tind}, i, newkey, oldlist, newlist) where {Tind}

    #logwrite(fn,"pq move: $(i), $(newkey), $(oldlist), $(newlist)")

    prev = pq.elems[i].prev
    next = pq.elems[i].next

    # remove i from its old list
    if next > zero(Tind)
        # really just modifying the first entry: prev
        pq.elems[next] = acPQ_elem{Tind}(prev, pq.elems[next].next, pq.elems[next].key)
    else
        # if this is the tail
        pq.tails[1+oldlist] = prev
    end

    if prev > zero(Tind)
        # really just modifying the second entry: next
        pq.elems[prev] = acPQ_elem{Tind}(pq.elems[prev].prev, next, pq.elems[prev].key)
    else
        # if this is the head
        pq.heads[1+oldlist] = next
    end

    # insert i into its new list
    tail = pq.tails[1+newlist]
    if tail > 0
        pq.elems[tail] = acPQ_elem{Tind}(pq.elems[tail].prev, i, pq.elems[tail].key)
    end
    pq.tails[1+newlist] = i

    pq.elems[i] = acPQ_elem{Tind}(tail, zero(Tind), newkey)

    if pq.heads[1+newlist] == 0
        pq.heads[1+newlist] = i
    end

    #return Void
end

"""
    Decrement the key of element i
    This could crash if i exceeds the maxkey
"""
function pqDec!(pq::acPQ_rev{Tind}, i) where {Tind}

    #logwrite(fn,"pq dec: $(i)")

    oldlist = keyMap(pq.elems[i].key, pq.n)
    newlist = keyMap(pq.elems[i].key - one(Tind), pq.n)

    if newlist != oldlist

        pqMove!(pq, i, pq.elems[i].key - one(Tind), oldlist, newlist)

        if newlist < pq.minlist
            pq.minlist = newlist
        end

    else
        pq.elems[i] = acPQ_elem{Tind}(pq.elems[i].prev, pq.elems[i].next, pq.elems[i].key - one(Tind))
    end

    #return Void
end

"""
    Increment the key of element i
    This could crash if i exceeds the maxkey
"""
function pqInc!(pq::acPQ_rev{Tind}, i) where {Tind}

    #logwrite(fn,"pq inc: $(i)")

    oldlist = keyMap(pq.elems[i].key, pq.n)
    newlist = keyMap(pq.elems[i].key + one(Tind), pq.n)

    if newlist != oldlist

        pqMove!(pq, i, pq.elems[i].key + one(Tind), oldlist, newlist)

    else
        pq.elems[i] = acPQ_elem{Tind}(pq.elems[i].prev, pq.elems[i].next, pq.elems[i].key + one(Tind))
    end

    #return Void
end
