
isdifferent(a,b) = a == b ? 0 : 1

const FitnessBreakpoint = 10.0
function pd(i1, i2, o1, o2;
    dist_output = isdifferent, #TODO in future allow for ordinal distance or any other measure.
    dist_input = Euclidean(),
    Delta = 1e-6)

    dout = dist_output(o1, o2)
    din = dist_input(i1, i2)

    if dout == 0.0
        return Inf # maximially penalize if same category - not our intention.
        #return FitnessBreakpoint + 1.0 / (din + Delta) # Push points away until we hopefully find a difference in the outputs
    else
        return FitnessBreakpoint - dout / (din + Delta) # When we have some output distance we start giving a benefit to points being close.
    end
end

nargs(f::Function) = length(methods(f).ms[1].sig.parameters)-1

#TODO fitness function does not match for iris somehow.
function pd_fitness(_classifier::Function; dist_output = isdifferent, Delta = 1e-6)
    return (x::Vector{Float64}) -> begin
        i1 = x[1:nargs(_classifier)]
        i2 = x[(nargs(_classifier)+1):end]
        o1 = _classifier(i1...)
        o2 = _classifier(i2...)
        pd(i1, i2, o1, o2; dist_output, Delta)
    end
end

struct BoundaryCandidate 
    i_left
    i_right

    function BoundaryCandidate(bbo_sol::Vector)
        l = trunc(Int64, length(bbo_sol) / 2)
        return new(bbo_sol[1:l], bbo_sol[(l+1):end])
    end
end

left(bc::BoundaryCandidate) = bc.i_left
right(bc::BoundaryCandidate) = bc.i_right

struct BoundarySqueeze
    ranges
    dist_input
    dist_output
    delta
    npoints

    BoundarySqueeze(ranges; Delta = 1e-6, npoints=20) = new(ranges, Euclidean(), isdifferent, Delta, npoints)
end

ranges(bs::BoundarySqueeze) = bs.ranges
distinput(bs::BoundarySqueeze) = bs.dist_input
distoutput(bs::BoundarySqueeze) = bs.dist_output
delta(bs::BoundarySqueeze) = bs.delta
npoints(bs::BoundarySqueeze) = bs.npoints

function convergence_callback(bs::BoundarySqueeze)
    return (c::BlackBoxOptim.OptRunController) ->
    begin
        bc = BoundaryCandidate(best_candidate(c))
        if best_fitness(c) < Inf # valid pair (boundary has candidate on two sides of the boundary)
            dist = distinput(bs)(left(bc), right(bc))
            
            if dist < delta(bs)
                    c.max_steps = BlackBoxOptim.num_steps(c)-1
                    println("--------")
                    println(c.max_steps)
                    println(best_fitness(c))
                    println(bc)
                    println("--------")
            end
        end
    end
end

# TODO create test case
function trivialpopulation(bs::BoundarySqueeze, cand1, cand2)
    seed = vcat(cand1, cand2)
    
    ncandinputs = length(cand1) * 2 # always same length

    # TODO must do the boxing - ensure in range
    pop = Matrix{Float64}(undef, ncandinputs, npoints(bs)-1)
    for i in 1:(npoints(bs)-1)
        pop[:, i] = seed .+ (rand(ncandinputs) .- 0.5) .* 2 * delta(bs) # introduce some random noise
    end

    return convert(Matrix{Float64}, [pop seed])
end

function apply(bs::BoundarySqueeze, sut::Function, cand1, cand2;
                                                        MaxTime=3::Int,
                                                        dist_output = isdifferent)
    Population = trivialpopulation(bs, cand1, cand2)
    Delta = delta(bs)
    _ranges = map(r -> (r[1]-Delta, r[2]+Delta), ranges(bs))
    res = bboptimize(pd_fitness(sut; dist_output); SearchRange = vcat(_ranges, _ranges),
                                CallbackInterval = 0.0,
                                CallbackFunction = convergence_callback(bs),
                                MaxTime, Population, Delta)

    return BoundaryCandidate(best_candidate(res))
end

# create example code for bmi, bmi_classification, synth, synth_bound, and iris
# TODO create a plotting function for the boundary candidates after boundaryexposer has been applied.
# TODO for all training data sets without sut, groundtruth or model, have a simple way to create models to the classification problem.