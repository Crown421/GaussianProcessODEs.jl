export getborders

# TODO: possibly make smarter
function getborders(y)
    [[minimum(getindex.(y.u, 1))-0.2, maximum(getindex.(y.u, 1))+0.2], [minimum(getindex.(y.u, 2))-0.2, maximum(getindex.(y.u, 2))+0.2] ]
end