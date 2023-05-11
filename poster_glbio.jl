include("init.jl")
include("data_processing.jl")
include("assoc_engines.jl")
include("utils.jl")
outpath, session_id = set_dirs()

### TCGA (all 33 cancer types)
tcga_prediction = GDC_data("Data/GDC_processed/TCGA_TPM_hv_subset.h5", log_transform = true, shuffled =true);
#tcga_prediction = GDC_data("Data/GDC_processed/TCGA_TPM_lab.h5", log_transform = true, shuffled =true);
abbrv = tcga_abbrv()
##### BRCA (5 subtypes)
brca_prediction= GDC_data("Data/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);


assoc_ae_params = Dict("dataset" => "tcga_prediction", "model_type" => "assoc_ae", "session_id" => session_id, 
"modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "nsamples" => length(tcga_prediction.rows),
"insize" => length(tcga_prediction.cols), "ngenes" => length(tcga_prediction.cols), "nclasses"=> length(unique(tcga_prediction.targets)), 
"nfolds" => 5,  "nepochs" => 100, "mb_size" => 4000, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-7, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)

params = assoc_ae_params
# init 
mkdir("RES/$(params["session_id"])/$(params["modelid"])")
# init results lists 
true_labs_list, pred_labs_list = [],[]
# create fold directories
[mkdir("RES/$(params["session_id"])/$(params["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params["nfolds"]]
# splitting, dumped 
folds = split_train_test(tcga_prediction.data, label_binarizer(tcga_prediction.targets), nfolds = params["nfolds"])
dump_folds(folds, params, tcga_prediction.rows)
# dump params
bson("RES/$(params["session_id"])/$(params["modelid"])/params.bson", params)
# define dump call back 
function dump_model_cb(dump_freq, targets)
    return (model, params,iter::Int, fold) -> begin 
        # check if end of epoch / start / end 
        if iter % dump_freq == 0 || iter == 1 || iter == params["nepochs"]
            # saves model
            #bson("RES/$(params["session_id"])/$(params["modelid"])/FOLD$(zpad(fold["foldn"],pad =3))/model_$(zpad(iter)).bson", model)
            # plot learning curve

            # plot embedding
            X_tr = cpu(model.ae.encoder(gpu(fold["train_x"]')))
            labels = tcga_abbrv(targets[fold["train_ids"]])
            fig_outpath = "RES/$(params["session_id"])/$(params["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))/model_$(zpad(iter)).svg"
            plot_embed(X_tr, labels, params, fig_outpath)
 
        end 
    end 
end 

dump_cb = dump_model_cb(1, tcga_prediction.targets)
model = build(params)
train_metrics = train!(model, folds[1], dump_cb, params)

# start crossval
for (foldn, fold) in enumerate(folds)
    model = build(model_params)
    train_metrics = train!(model, fold, dump_cb, nepochs = model_params["nepochs"], batchsize = model_params["mb_size"], wd = model_params["wd"])
    true_labs, pred_labs = test(model, fold)
    push!(true_labs_list, true_labs)
    push!(pred_labs_list, pred_labs)
    println("train: ", train_metrics)
    println("test: ", accuracy(true_labs, pred_labs))
    # post run 
    # save model
    # save 2d embed svg
    # training curves svg, csv 
end
### bootstrap results get 95% conf. interval 
low_ci, med, upp_ci = bootstrap(accuracy, true_labs_list, pred_labs_list) 
### returns a dict 
ret_dict = Dict("cv_acc_low_ci" => low_ci,
"cv_acc_upp_ci" => upp_ci,
"cv_acc_median" => med
)
model_params["cv_acc_low_ci"] = low_ci
model_params["cv_acc_median"] = med
model_params["cv_acc_upp_ci"] = upp_ci
# param dict 
return ret_dict

function validate!()

end 



### dimensionality reductions 
model = build(assoc_ae_params)
tcga_ae_red, tr_metrics = fit_transform!(model, tcga_prediction, assoc_ae_params);
assoc_ae_params["tr_acc"] = accuracy(gpu(label_binarizer(tcga_prediction.targets)'), model.clf.model(gpu(tcga_prediction.data')))
function plot_embed(X_tr, labels, assoc_ae_params,fig_outpath)
    # plot final 2d embed from Auto-Encoder
    tr_acc = round(assoc_ae_params["tr_acc"], digits = 3) * 100
    embed = DataFrame(:emb1=>X_tr[1,:], :emb2=>X_tr[2,:], :cancer_type => labels)
    p = AlgebraOfGraphics.data(embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type)
    fig = draw(p, axis = (;width = 1224, height = 1024, 
    title="$(assoc_ae_params["model_type"]) on $(assoc_ae_params["dataset"]) data\naccuracy by DNN : $tr_acc%"))
    CairoMakie.save(fig_outpath, fig)
end 

function stringify(p::Dict;spacer = 80)  
    s = join(["$key: $val" for (key, val) in p], ", ")
    for i in collect(spacer:spacer:length(s))
        s = "$(s[1:i])\n$(s[i:end])"
    end
    return s 
end 

function plot_learning_curves(learning_curves, assoc_ae_params)
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), :ae_loss=>[i[1] for i in learning_curves], :ae_cor => [i[2] for i in learning_curves],
    :clf_loss=>[i[3] for i in learning_curves], :clf_acc => [i[4] for i in learning_curves])
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"ae_loss"], color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "Classifier Crossentropy loss")
    ae_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"clf_loss"])
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder Pearson Corr.")
    ae_loss = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"ae_cor"], color = "red")
    fig[2,2] = Axis(fig, xlabel = "steps", ylabel = "Classfier Accuracy (%)")
    ae_loss = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"clf_acc"] .* 100 )
    Label(fig[3,:], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(assoc_ae_params))")
    CairoMakie.save("RES/$(assoc_ae_params["session_id"])/$(assoc_ae_params["model_type"])_$(assoc_ae_params["dataset"])_lrn_curve_$(assoc_ae_params["modelid"]).svg", fig)
end 
plot_learning_curves(tr_metrics, assoc_ae_params)
 
tcga_prediction_assoc_ae_res = validate(assoc_ae_params, tcga_prediction;nfolds = assoc_ae_params["nfolds"])
bson("$outpath/tcga_prediction_assoc_ae_params.bson", assoc_ae_params)



