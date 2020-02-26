using Plots
pyplot()

export plotsolutions

function plotsolutions(refsol, compsol)
    p = plot(layout = (1,2), size = (1100, 600))

    for (key, sol) in compsol
        plot!(p, sol, vars = (1,2), label = key, subplot = 1, linewidth = 3)
    end
    plot!(p, refsol, vars = (1,2), label = "reference", color = :black, linestyle = :dash,
        subplot = 1, title = "state space")


    interpt = refsol.t[1]:0.1:refsol.t[end]
    rs = reduce(hcat, refsol(interpt))

    for (key, sol) in compsol
        cs = reduce(hcat, sol(interpt))
        err = sum(abs.(cs - rs), dims = 1)[:]
        println("$key: $(sum(err)/length(err))")

        plot!(p, interpt, err, label = key, subplot = 2, title = "error")
    end
    p
end