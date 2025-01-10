wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth # SAM VIT-B
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth # SAM VIT-L
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth # SAM VIT-H (default)

# mmdet style SAM ViT backbone weights
wget -nc https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth -O mmdet_sam_vit_b_01ec64_backbone.pth
wget -nc https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-large-p16_sam-pre_3rdparty_sa1b-1024px_20230411-595feafd.pth -O mmdet_sam_vit_l_0b3195_backbone.pth
wget -nc https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-huge-p16_sam-pre_3rdparty_sa1b-1024px_20230411-3f13c653.pth -O mmdet_sam_vit_h_4b8939_backbone.pth

# mmdet-style state dict for weight conversion
wget -nc https://seafile.unistra.fr/f/06ab103c19cb43439ce0/?dl=1 -O ref/mmdet_sam_vit_b_01ec64_state_dict.pth
if [ ! -e "ref/mmdet_medsam_vit_b_state_dict.pth" ]; then
    ln -s mmdet_sam_vit_b_01ec64_state_dict.pth ref/mmdet_medsam_vit_b_state_dict.pth
fi

# medsam
if [ ! -e "medsam_vit_b.pth" ]; then
    gdown https://drive.google.com/uc?id=1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_
fi

# sam-med2d
if [ ! -e "sam-med2d_b.pth" ]; then
    gdown https://drive.google.com/uc?id=1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl
fi
