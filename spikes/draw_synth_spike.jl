# A script that graphically shows the progress over the ML squeeze search.
# first outputs png files, and then stitches a video out of it (running external script create_synth_video.sh)

using CSV, DataFrames, StatsPlots

synth_func(x) = (x+2)*(x^2-4)

# expected data structure in CSV
# Ax,Ay,Bx,By,candA1x,candA1y,candB1x,candB1y,...,candANx,candANy,candBNx,candBNy,fit1,fit2,fit3
df = CSV.read(joinpath(@__DIR__, "..", "data", "synth_boundary_set_w5_w1.csv"), DataFrame)

res_dir = joinpath("data","synth_run")

rm(res_dir, force = true, recursive = true)
mkpath(res_dir)
num_cands = filter(n -> !isnothing(match(r"^candA\d*x", n)), names(df)) |> length
cand_names_Ax = map(n -> "candA$(n)x", 1:num_cands)
cand_names_Ay = map(n -> "candA$(n)y", 1:num_cands)
cand_names_Bx = map(n -> "candB$(n)x", 1:num_cands)
cand_names_By = map(n -> "candB$(n)y", 1:num_cands)

for (i, iter_row) in enumerate(eachrow(df))
    a_xs = iter_row[cand_names_Ax] |> collect
    a_ys = iter_row[cand_names_Ay] |> collect
    b_xs = iter_row[cand_names_Bx] |> collect
    b_ys = iter_row[cand_names_By] |> collect
    a_start_x = iter_row["Ax"]
    a_start_y = iter_row["Ay"]
    b_start_x = iter_row["Bx"]
    b_start_y = iter_row["By"]

    xs = vcat(a_xs, b_xs, [a_start_x, b_start_x])
    xlims = extrema(xs)
    ys = vcat(a_ys, b_ys, [a_start_y, b_start_y])
    ylims = extrema(ys)

    xmargin = (xlims[2] - xlims[1]) * .1
    xlims = (xlims[1] - xmargin, xlims[2] + xmargin)

    ymargin = (ylims[2] - ylims[1]) * .1
    ylims = (ylims[1] - ymargin, ylims[2] + ymargin)

    # instantiate the synthetic boundary
    x = range(xlims[1], xlims[2], 1_000)
    y = synth_func.(x)

    p = plot(x, y, xlims=xlims, ylims=ylims, label = "boundary")

    plot!((a_start_x, a_start_y), seriestype=:scatter, label = "startpoint A", color = :green, markershape = :rect)
    plot!((b_start_x, b_start_y), seriestype=:scatter, label = "startpoint B", color = :red, markershape = :rect)
    plot!(a_xs, a_ys, seriestype=:scatter, label = "a candidates", color = :green)
    plot!(b_xs, b_ys, seriestype=:scatter, label = "b candidates", color = :red)

    for n in 1:num_cands
        plot!([(a_xs[n], a_ys[n]),(b_xs[n], b_ys[n])], label="", line=(nothing, :black, :solid))
    end

    title!("Boundary Squeeze on Synth Classification Problem")
    xlabel!("dim 1")
    ylabel!("dim 2")

    text_x = xlims[1] + (xlims[2] - xlims[1]) * .9
    text_y = ylims[1] + (ylims[2] - ylims[1]) * .1
    annotate!((text_x,text_y,text("fitness: $(iter_row.fitness3)", 14, :right, :bottom)))
    _digits = digits(i) |> length
    i_fig = "$(repeat("0", 5-_digits))$i"
    savefig(p, joinpath(res_dir, "fig_$(i_fig).png"))
    print(".")
end

# create video with something like:
# ffmpeg -framerate 10 -i fig_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4