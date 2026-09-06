

datasets=(  
    # hest/bench_data/CCRCC
    # hest/bench_data/PRAD 
    # hest/kidney  
    nameeta/GBM
    # takano/xenium
    # hest/andrew
    # hest/andersson
    # hest/bryan
    takano/visium

)

configs=(
    # hest/CCRCC
    # hest/PRAD
    # hest/kidney
    nameeta/GBM
    # takano/xenium
    # ST/andrew
    # ST/andersson
    # ST/bryan
    takano/visium
)

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    config="${configs[$i]}"
    # echo "Processing dataset: ${dataset} with config: ${config}"
    # echo /home/shared/chungym/project/STpredBench/logs/bench/${config}/LocalNet

    python preprocess.py \
        --dataset_path "/home/shared/chungym/hier_st/input/${dataset}" \
        --ckpt_path "/home/chungym/project/STpredBench/logs/bench/${config}/LocalNet" \
        --mode "train"
done

# python preprocess.py \
#     --dataset_path "/home/shared/chungym/hier_st/input/takano/xenium" \
#     --ckpt_path "/home/chungym/project/STpredBench/logs/bench/takano/xenium/LocalNet" \
#     --mode "train"

# for dataset in "${datasets[@]}"; do
#     python preprocess.py \
#         --dataset_path "/home/shared/chungym/hier_st/input/${dataset}" \
#         --ckpt_path "/home/chungym/project/STpredBench/logs/bench/${dataset}/LocalNet" \
#         --mode "train"
# done


# dataset="hest/bench_data/CCRCC"
# config="hest/CCRCC"

# python preprocess.py \
#     --dataset_path "/home/shared/chungym/hier_st/input/${dataset}" \
#     --ckpt_path "/home/chungym/project/STpredBench/logs/bench/${config}/LocalNet" \
#     --mode "train"


# dataset="hest/bench_data/PRAD"
# config="hest/PRAD"

# python preprocess.py \
#     --dataset_path "/home/shared/chungym/hier_st/input/${dataset}" \
#     --ckpt_path "/home/chungym/project/STpredBench/logs/bench/${config}/LocalNet" \
#     --mode "train"