##############################################
# GENERAL
##############################################

experiment_name = 'gtloc'
seed = 34


##############################################
# DATASETS
##############################################

# Adjust paths as necessary
mp16_metadata_path = '../metadata/mp16.csv'
cvt_metadata_path = '../metadata/cvt.csv'
skyfinder_train_metadata_path = '../metadata/skyfinder_train.csv'
skyfinder_val_metadata_path = '../metadata/skyfinder_val.csv'
im2gps3k_metadata_path = '../metadata/im2gps3k.csv'
gws15k_metadata_path = '../metadata/gws15k.csv'

mp16_imgs_path = '/home/c3-0/datasets/MP-16/resources/images/mp16'
cvt_imgs_path = '/home/da625117/wriva/datasets/CVT/cvt_512/images/ground/yfcc100m_phone'
skyfinder_imgs_path = '/home/da625117/wriva/datasets/CVT/cvt_512/images/ground/amos'
im2gps3k_imgs_path = '/home/al209167/datasets/im2gps3ktest'
gws15k_imgs_path = '/home/c3-0/al209167/datasets/GS10k'

tencrop = True
galleries = 'data_dist' # 'data_dist', 'random'
dataset_sample = 1.0


##############################################
# MODEL
##############################################

backbone = 'clip'
freeze_backbone = True

time_sigma = [2**0, 2**4, 2**8]
loc_sigma = [2**0, 2**4, 2**8]

queue_size = 4096
hidden_dim = 768
embedding_dim = 512


##############################################
# EVAL
##############################################

eval_bsz = 512
num_workers = 12