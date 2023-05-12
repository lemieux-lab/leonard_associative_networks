function get_GDC_CLIN_data_init_paths()

    # loading data
    CLIN_FULL = CSV.read("Data/GDC_clinical_raw.tsv", DataFrame)
    # MANIFEST = CSV.read("data/gdc_manifest_GE_2023-02-02.txt", DataFrame)
    # IDS = CSV.read("data/sample.tsv", DataFrame)
    baseurl = "https://api.gdc.cancer.gov/data"
    basepath = "Data/DATA/GDC_processed"
    FILES = "$basepath/GDC_files.json"
    J = JSON.parsefile(FILES)
    features = ["case_id", "case_submitter_id", "project_id", "gender", "age_at_index","age_at_diagnosis", "days_to_death", "days_to_last_follow_up", "primary_diagnosis", "treatment_type"]
    CLIN = CLIN_FULL[:, features]
    return J, CLIN_FULL, CLIN, baseurl, basepath
end 

function generate_fetch_data_file(J, baseurl, basepath)
    outputfile = "$basepath/fetch_data.sh"
    f = open(outputfile, "w")
    ## FECTHING DATA 
    for i::Int in ProgressBar(1:length(J))
        file_id = J[i]["file_id"]
        # println(file_id)
        case_id = J[i]["cases"][1]["case_id"]
        # println(case_id)
        cmd = "curl $baseurl/$file_id -o $basepath/GDC/$case_id\n"
        write(f, cmd)
        # cmd = `curl $baseurl/$file_id -o $basepath/$case_id`
        #run(cmd)
    end 
    close(f)
end 

struct GDC_data
    data::Matrix
    rows::Array 
    cols::Array
    targets::Array
end 
function GDC_data(inputfile::String; log_transform=false, shuffled = true)
    tpm, cases, gnames, labels = load_GDC_data(inputfile;log_transform=log_transform, shuffled = shuffled)
    return GDC_data(tpm, cases, gnames, labels)
end
function load_GDC_data(infile; log_transform = false, shuffled= true)
    inf = h5open(infile, "r")
    tpm_data = inf["data"][:,:]
    case_ids = inf["rows"][:]
    gene_names = inf["cols"][:] 
    if in("labels", keys(inf))
        labels = inf["labels"][:]
    else 
        labels = zeros(length(case_ids))
    end 

    close(inf)
    if log_transform
        tpm_data = log10.(tpm_data .+1 )
    end 
    
    ids = collect(1:length(case_ids))
    shuffled_ids = shuffle(ids)
    if shuffled
        ids = shuffled_ids
    end 
    return tpm_data[ids,:], case_ids[ids], gene_names, labels[ids]
end   


function write_h5(dat::GDC_data, labels, outfile)
    # HDF5
    # writing to hdf5 
    f = h5open(outfile, "w")
    f["data"] = dat.data
    f["rows"] = dat.rows
    f["cols"] = dat.cols
    f["labels"] = labels 
    close(f)
end 

function merge_GDC_data(basepath, outfile)
    files = readdir(basepath)
    sample_data = CSV.read("$basepath/$(files[1])", DataFrame, delim = "\t", header = 2)
    sample_data = sample_data[5:end, ["gene_name", "tpm_unstranded"]]
    nsamples = length(files)
    ngenes = size(sample_data)[1]
    m=Array{Float32, 2}(undef, (nsamples, ngenes))

    for fid::Int in ProgressBar(1:length(files))
        file = files[fid]
        dat = CSV.read("$basepath/$(file)", DataFrame, delim = "\t", header = 2)
        dat = dat[5:end, ["gene_name", "tpm_unstranded"]]
        m[fid, :] = dat.tpm_unstranded
    end
    output_data = GDC_data(m, files, Array{String}(sample_data.gene_name)) 
    write_h5(output_data, outfile)
    return output_data 
end 

function tcga_abbrv()
    abbrv = CSV.read("Data/GDC_processed/TCGA_abbrev.txt", DataFrame, delim = ",")
    abbrvDict = Dict([("TCGA-$(String(strip(abbrv[i,1])))", abbrv[i,2]) for i in 1:size(abbrv)[1]])
    return abbrvDict
end
function tcga_abbrv(targets::Array)
    abbrv = CSV.read("Data/GDC_processed/TCGA_abbrev.txt", DataFrame, delim = ",")
    abbrvDict = Dict([("TCGA-$(String(strip(abbrv[i,1])))", abbrv[i,2]) for i in 1:size(abbrv)[1]])
    return [abbrvDict[l] for l in targets]
end

function preprocess_data(GDCd, CLIN, outfilename; var_frac = 0.75)
    cases = GDCd.rows
    ngenes = length(GDCd.cols)
    # intersect with clin data 
    uniq_case_id = unique(CLIN.case_id)
    keep = [in(c,uniq_case_id ) for c in cases]
    GDC = GDC_data(GDCd.data[keep,:], GDCd.rows[keep], GDCd.cols)

    # map to tissues 
    cid_pid = Dict([(cid, pid) for (cid, pid) in zip(CLIN.case_id, Array{String}(CLIN.project_id))])
    tissues = [cid_pid[c] for c in GDC.rows]
    # filter on variance
    vars = vec(var(GDC.data, dims = 1))  
    hv = vec(var(GDC.data, dims =1 )) .> sort(vars)[Int(round(var_frac * ngenes))]

    GDC_hv = GDC_data(GDC.data[:,hv], GDC.rows, GDC.cols[hv])

    f = h5open(outfilename, "w")
    f["data"] = GDC_hv.data
    f["rows"] = GDC_hv.rows
    f["cols"] = GDC_hv.cols
    f["tissues"] = tissues
    close(f)
    return GDC_hv
end 

function run_tsne_on_GDC(GDC_data, tissues)
    @time TCGA_tsne = tsne(GDC_data, 2, 50, 1000, 30.0;verbose=true,progress=true)

    TSNE_df = DataFrame(Dict("dim_1" => TCGA_tsne[:,1], "dim_2" => TCGA_tsne[:,2], "tissue" => tissues))

    q = AlgebraOfGraphics.data(TSNE_df) * mapping(:dim_1, :dim_2, color = :tissue, marker = :tissue) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)

    main_fig = draw(q ; axis=(width=1024, height=1024,
                    title = "2D TSNE by tissue type on GDC data, number of input genes: $(size(tcga_hv.data)[2]), nb. samples: $(size(tcga_hv.data)[1])",
                    xlabel = "TSNE 1",
                    ylabel = "TSNE 2"))


    save("RES/GDC_$(nsamples)_samples.svg", main_fig, pt_per_unit = 2)

    save("RES/GDC_$(nsamples)_samples.png", main_fig, pt_per_unit = 2)

end 

