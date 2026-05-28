
datasets=(  
    # hest/bench_data/CCRCC
    # hest/bench_data/PRAD 
    # hest/kidney  
    nameeta/GBM
    # takano/xenium
    takano/visium
    # ST/andrew
    # ST/andersson
    # ST/bryan
)

for dataset in "${datasets[@]}"; do
    python build_exemplar.py \
        --data_dir "/home/shared/chungym/hier_st/input/${dataset}" \
        --distance_metric l1

    python build_exemplar.py \
        --data_dir "/home/shared/chungym/hier_st/input/${dataset}" \
        --distance_metric l2
done

# python build_exemplar.py --data_dir /home/shared/chungym/hier_st/input/takano/xenium --distance_metric l1