# module Init
using Pkg
Pkg.activate(".")

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
using SHA
using BSON 

function set_dirs()
    session_id = "$(now())"
    outpath = "./RES/$session_id"
    mkdir(outpath)

    return outpath, session_id
end

# end 