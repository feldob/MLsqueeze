# A script that graphically shows the progress over the ML squeeze search.
# first outputs png files, and then stitches a video out of it (running external script create_synth_video.sh)

using CSV, DataFrames, StatsPlots

# instantiate the synthetic boundary
synth_func(x) = (x+2)*(x^2-4)
x = range(-10, 10, 1_000)
y = synth_func.(x)

# expected data structure in CSV
# Ax,Ay,Bx,By,candA1x,candA1y,candB1x,candB1y,...,candANx,candANy,candBNx,candBNy,fit1,fit2,fit3
df = CSV.read("data/synth_run.csv", DataFrame)

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

    # plot boundary, starting points for search, 
    p = plot(x, y, xlims=(-10, 10), ylims=(-10,10), label = "boundary")
    plot!((a_start_x, a_start_y), seriestype=:scatter, label = "startpoint A", color = :green, markershape = :rect)
    plot!((b_start_x, b_start_y), seriestype=:scatter, label = "startpoint B", color = :red, markershape = :rect)
    plot!(a_xs, a_ys, seriestype=:scatter, label = "a candidates", color = :green)
    plot!(b_xs, b_ys, seriestype=:scatter, label = "b candidates", color = :red)
    title!("Boundary Squeeze on Synth Classification Problem")
    xlabel!("dim 1")
    ylabel!("dim 2")
    annotate!((8,-8,text("fitness: $(iter_row.fitness)", 14, :right, :bottom)))
    _digits = digits(i) |> length
    i_fig = "$(repeat("0", 5-_digits))$i"
    savefig(p, joinpath(res_dir, "fig_$(i_fig).png"))
end