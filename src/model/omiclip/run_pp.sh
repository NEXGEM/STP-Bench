
test_datasets=(
    # hest/bench_data/IDC
    # hest/bench_data/LUNG
    # hest/bench_data/PAAD
    # hest/GBM
    # wustl/RCC
    massey/TNBC
)

fold=0
for dataset in "${test_datasets[@]}"; do
    python preprocess/cal_similarity_matrix.py --external_data $dataset --fold $fold
done