include("init.jl")
include("data_processing.jl")
include("assoc_engines.jl")
include("utils.jl")
#device!()
outpath, session_id = set_dirs()

### TCGA (all 33 cancer types)
tcga_prediction = GDC_data("Data/GDC_processed/TCGA_TPM_hv_subset.h5", log_transform = true, shuffled =true);
#tcga_prediction = GDC_data("Data/GDC_processed/TCGA_TPM_lab.h5", log_transform = true, shuffled =true);
abbrv = tcga_abbrv()
##### BRCA (5 subtypes)
brca_prediction= GDC_data("Data/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);


assoc_ae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "tcga_prediction", 
"model_type" => "assoc_ae", "session_id" => session_id, "nsamples" => length(tcga_prediction.rows),
"insize" => length(tcga_prediction.cols), "ngenes" => length(tcga_prediction.cols), "nclasses"=> length(unique(tcga_prediction.targets)), 
"nfolds" => 5,  "nepochs" => 20_000, "mb_size" => 1000, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-7, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)

dump_cb_dev = dump_model_cb(Int(floor(assoc_ae_params["nsamples"] / assoc_ae_params["mb_size"])), tcga_prediction.targets)


brca_mtae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "assoc_ae", "session_id" => session_id, "nsamples" => length(brca_prediction.rows),
"insize" => length(brca_prediction.cols), "ngenes" => length(brca_prediction.cols), "nclasses"=> length(unique(brca_prediction.targets)), 
"nfolds" => 5,  "nepochs" => 5_000, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-3, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)

validate!(brca_mtae_params, brca_prediction, dummy_dump_cb)
# validate!(assoc_ae_params, tcga_prediction, dump_cb_dev)

### dimensionality reductions 
model = build(assoc_ae_params)
tcga_ae_red, tr_metrics = fit_transform!(model, tcga_prediction, assoc_ae_params);
assoc_ae_params["tr_acc"] = accuracy(gpu(label_binarizer(tcga_prediction.targets)'), model.clf.model(gpu(tcga_prediction.data')))

lr_fig_outpath = "RES/$(params["session_id"])/$(params["model_type"])/FOLD($(zpad(foldn))_lr_curve.svg"
plot_learning_curves(tr_metrics, assoc_ae_params, lr_fig_outpath)
 
tcga_prediction_assoc_ae_res = validate(assoc_ae_params, tcga_prediction;nfolds = assoc_ae_params["nfolds"])
bson("$outpath/tcga_prediction_assoc_ae_params.bson", assoc_ae_params)



