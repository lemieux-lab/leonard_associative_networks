# module Init
using Pkg
Pkg.activate("/u/sauves/leonard_leucegene_factorized_embeddings/")

# imports  
using CSV
using DataFrames
using JSON
using ProgressBars
using HDF5
using Statistics
using Flux 
using CairoMakie 
using Random
using Dates
using AlgebraOfGraphics
using JuBox
using CUDA

function set_dirs()
    session_id = "$(now())"
    outpath = "./RES/EMBEDDINGS/$session_id"
    mkdir(outpath)

    return outpath, session_id
end

# end 