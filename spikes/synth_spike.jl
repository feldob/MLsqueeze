using Plots

synth_func(x) = (x+2)*(x^2-4)

x = range(-10, 10, 100_000)
y = synth_func.(x)

plot(x, y, xlims=(-10, 10), ylims=(-10,10), label = "boundary")

randinrange(n::Integer) = -10 .+ rand(n) .* 20

x = randinrange(200)
y = randinrange(200)

fx = synth_func.(x)

x1, y1, x2, y2 = [], [], [], []

for i in eachindex(x)
    if y[i] > fx[i]
        push!(x1, x[i])
        push!(y1, y[i])
    else
        push!(x2, x[i])
        push!(y2, y[i])
    end
end

plot!(x1,y1, seriestype=:scatter, label = "class 1")
plot!(x2,y2, seriestype=:scatter, label = "class 2")

# png("data/synth_class.png")