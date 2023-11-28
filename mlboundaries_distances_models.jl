using CSV, DataFrames ,MLsqueeze, DecisionTree

mutable struct ExperimentPair
    id
    df_div
    df_rand
    inputs

    function ExperimentPair(id, div, rand, inputs)
        df_div = CSV.read(div, DataFrame)
        df_rand = CSV.read(rand, DataFrame)
        new(id, df_div, df_rand, inputs)
    end
end

calc_div(ep::ExperimentPair) = two_nearest_neighbor_distances(ep.df_div, ep.inputs)
function calc_rand(ep::ExperimentPair)   
    @assert nrow(ep.df_div) ≤ nrow(ep.df_rand)
    
    if nrow(ep.df_rand) > nrow(ep.df_div)
        ep.df_rand = ep.df_rand[1:nrow(ep.df_div), :]
    end

    return two_nearest_neighbor_distances(ep.df_rand, ep.inputs)
end

synth_exp = ExperimentPair("synth", "data/synth_model_points_div.csv", "data/synth_model_points_random.csv", [:x , :y])
iris_exp =  ExperimentPair("iris", "data/iris_bcs_div.csv", "data/iris_bcs_random.csv", [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm])
titanic_exp =  ExperimentPair("titanic", "data/titanic_bcs_postprocessed.csv", "data/titanic_bcs_raw_random.csv", [:Pclass, :Age, :Sex, :Fare, :Parch, :SibSp])

# need to convert one side (left side) of the boundary, because only that side is used for calculation of distance
titanic_exp.df_div.Sex = map(d -> d == "male" ? 0.0 : 1.0, titanic_exp.df_div.Sex)

titanic_exp.df_rand.Pclass = round.(Int, titanic_exp.df_rand.Pclass)
titanic_exp.df_rand.Age = round.(Int, titanic_exp.df_rand.Age)
titanic_exp.df_rand.Parch = round.(Int, titanic_exp.df_rand.Parch)
titanic_exp.df_rand.Fare = round.(titanic_exp.df_rand.Fare, digits=2)
titanic_exp.df_rand.SibSp = round.(Int, titanic_exp.df_rand.SibSp)
titanic_exp.df_rand.Sex = round.(titanic_exp.df_rand.Sex)

exps = [synth_exp, iris_exp, titanic_exp]

for exp in exps
    d_rand_nn_mean, d_rand_nn_sd = calc_rand(exp)
    d_div_nn_mean, d_div_nn_sd = calc_div(exp)
    
    println("SUT - $(exp.id)")
    println("diversity: $(round(d_div_nn_mean, digits=2)) ± $(round(d_div_nn_sd, digits=2))")
    println("random: $(round(d_rand_nn_mean, digits=2)) ± $(round(d_rand_nn_sd, digits=2))")
end