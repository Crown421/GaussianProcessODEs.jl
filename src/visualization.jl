# using Plots
# pyplot()

# export plotsolutions

# function plotsolutions(refsol, compsol)
#     p = plot(layout = (1,2), size = (1100, 600))

#     for (key, sol) in compsol
#         plot!(p, sol, vars = (1,2), label = key, subplot = 1, linewidth = 3)
#     end
#     plot!(p, refsol, vars = (1,2), label = "reference", color = :black, linestyle = :dash,
#         subplot = 1, title = "state space")


#     interpt = refsol.t[1]:0.1:refsol.t[end]
#     rs = reduce(hcat, refsol(interpt))

#     for (key, sol) in compsol
#         cs = reduce(hcat, sol(interpt))
#         err = sum(abs.(cs - rs), dims = 1)[:]
#         println("$key: $(sum(err)/length(err))")

#         plot!(p, interpt, err, label = key, subplot = 2, title = "error")
#     end
#     p
# end



# function plotsol(sols::NTuple{N, ODESolution}, names, vars) where N
#     sol = sols[1]
#     colors = ([:royalblue3 :deepskyblue2], [:firebrick :salmon1], [:darkgoldenrod :mediumseagreen])
    
#     p = plot(size = (1100, 450), layout = (1,2))
#     for i in 1:length(sols)
#         plot!(p, sols[i]; color = colors[i], linewidth = 2, label = names[i] .* string.(vars), subplot = 1)
#     end 
# end