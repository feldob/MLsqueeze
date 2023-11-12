module MLsqueeze

using CSV,
        DataFrames,
        StatsPlots,
        Combinatorics,
        Distances,
        BlackBoxOptim,
        StatsBase,
        DecisionTree, # testing only
        MLJ,
        MLJDecisionTreeInterface

export
        # suts
        bmi,
        bmi_classification,
        BMI_RANGES,
        check_synth_valid,

        defaultranges,
        TrainingData,
        deriveranges,
        ranges,
        tofile,
        npoints,
        sutname,
        classproblem,
        plots,
        ninputs,
        unique_outputs,
        inputcols,
        outputcol,
        getmodelsut,
        
        BoundaryCandidate,
        BoundarySqueeze,
        delta,
        left,
        right,
        apply,
        todataframe,

        BoundaryExposer

include("suts/bmi.jl")
include("suts/synth.jl")
include("suts/classifier.jl")
include("trainingdata.jl")
include("boundarysqueeze.jl")
include("boundaryexposer.jl")

end # module MLsqueeze
