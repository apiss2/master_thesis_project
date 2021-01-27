python cnngeometric-train.py config/cnngeometric.ini --config_type dlmachine_MR
python cnngeometric-train.py config/cnngeometric.ini --config_type dlmachine_CT
python cnngeometric_multidomain-train.py config/multi_cnngeometric.ini --config_type dlmachine_MRCT

python cnngeometric-train.py config/DANN_cnngeometric.ini --config_type dlmachine_test
python cnngeometric-train.py config/WDGR_cnngeometric.ini --config_type dlmachine_test
python cnngeometric-train.py config/MCDDA_cnngeometric.ini --config_type dlmachine_test

python segmentation-train.py config/DANN_segmentation.ini --config_type dlmachine_test
python segmentation-train.py config/WDGR_segmentation.ini --config_type dlmachine_test
python segmentation-train.py config/MCDDA_segmentation.ini --config_type dlmachine_test

python segmentation-train.py config/DANN_segmentation.ini --config_type save_encoder
python segmentation-train.py config/WDGR_segmentation.ini --config_type save_encoder

python cnngeometric_multidomain-train.py config/multi_cnngeometric.ini --config_type dlmachine_pretrained_DANN
python cnngeometric_multidomain-train.py config/multi_cnngeometric.ini --config_type dlmachine_pretrained_WDGR
python cnngeometric_multidomain-train.py config/multi_cnngeometric.ini --config_type dlmachine_pretrained_MCDDA
