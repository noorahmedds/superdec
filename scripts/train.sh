# Unzip shapenet into TMDIR
unzip "$WORK"/datasets/dataset_small_v1.1.zip '*pointcloud.npz' -d $TMPDIR
unzip "$WORK"/datasets/dataset_small_v1.1.zip '*.lst' -d $TMPDIR