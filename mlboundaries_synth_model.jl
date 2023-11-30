using CSV, DataFrames

# change numbers to get boundaries of varying density

output = :class
ranges = [(-10,10), (-10,10)]

using MLsqueeze

td = TrainingData(check_synth_valid; ranges, npoints = 100)

using DecisionTree # with model
modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=7), fit=DecisionTree.fit!)
be = BoundaryExposer(td, modelsut) # instantiate search alg

#high resolution DT boundary
candidates = apply(be; iterations=10000, initial_candidates, optimizefordiversity=false) # search and collect candidates
df_high_res = todataframe(candidates, modelsut; output)

# diversity
candidates = apply(be; iterations=2000, initial_candidates=20, add_new=false) # search and collect candidates
df_model_diversity_20 = todataframe(candidates, modelsut; output)
CSV.write("data/synth_model_points_div_20.csv", df_model_diversity_20)

candidates = apply(be; iterations=2000, initial_candidates=10, add_new=false) # search and collect candidates
df_model_diversity_10 = todataframe(candidates, modelsut; output)
CSV.write("data/synth_model_points_div_10.csv", df_model_diversity_10)

# b) random
candidates = apply(be; iterations=20, initial_candidates=20, optimizefordiversity=false) # search and collect candidates
df_model_random_20 = todataframe(candidates, modelsut; output)
CSV.write("data/synth_model_points_random_20.csv", df_model_random_20)

candidates = apply(be; iterations=10, initial_candidates=10, optimizefordiversity=false) # search and collect candidates
df_model_random_10 = todataframe(candidates, modelsut; output)
CSV.write("data/synth_model_points_random_10.csv", df_model_random_10)
