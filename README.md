# **UltraSam: A Foundation Model for Ultrasound using Large Open-Access Segmentation Datasets**

_Adrien Meyer, Aditya Murali, Farahdiba Zarin, Didier Mutter, Nicolas Padoy_

[![arXiv](https://img.shields.io/badge/arxiv-2307.15220-red)](https://arxiv.org/pdf/2411.16222) [![IJCARS](https://img.shields.io/badge/IJCARS-paper-blue)](https://link.springer.com/article/10.1007/s11548-025-03517-8)


![UltraSam](./assets/UltraSam_main.png)


## Minimal working example
<details>
<summary>Click to expand Install</summary>
This example guide you to download and use UltraSam in inference mode in a sample dataset.
The sample dataset, coco-based, is in "./sample_dataset" (using MMOTU2D samples).

Clone the repo
```bash
git clone https://github.com/CAMMA-public/UltraSam
cd UltraSam
```

Create a conda environment and activate it. (Tested with cuda-11.8 & gcc-12)
```bash
conda create --name UltraSam python=3.8 -y
conda activate UltraSam
```

Install the OpenMMLab suite and other dependencies
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
mim install mmpretrain
pip install tensorboard
```

Download UltraSam ckpt
```bash
wget -O ./UltraSam.pth "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"
export PYTHONPATH=\$PYTHONPATH:.
```

```bash
mim test mmdet configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py --checkpoint UltraSam.pth --cfg-options test_dataloader.dataset.data_root="sample_dataset" test_dataloader.dataset.ann_file="sample_coco_MMOTU2D.json" test_dataloader.dataset.data_prefix.img="sample_images" test_evaluator.ann_file="sample_dataset/sample_coco_MMOTU2D.json"  --work-dir ./work_dir/example --show-dir ./show_dir
```

It will run inference on the specified sample dataset, modified inline from the base config. Predicted mask are visible in the show-dir. That is it!

</details>


## Usage

<details>
<summary>Click to expand Install</summary>
You may need to install a specific version of PyTorch, depending on your hardware.

Create a conda environment and activate it.
```bash
conda create --name UltraSam python=3.8 -y
conda activate UltraSam
```

Install the OpenMMLab suite and other dependencies
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
mim install mmpretrain
```

If you wish to process the datasets;
```bash
pip install SimpleITK
pip install scikit-image
pip install scipy
```

Pre-trained UltraSam model checkpoint is accessible [at this link](https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth).

To train / test, you will need a coco.json annotation file, and create a symbolik link to it, or modify the config files to point to your annotation file.

To train from scratch, you can use the code in ```weights``` to download and convert SAM, MEDSAM and adapters weights.

In local, inside UltraSam repo;
```bash
export PYTHONPATH=$PYTHONPATH:.

mim train mmdet configs/UltraSAM/UltraSAM_full/UltraSAM_point_refine.py --gpus 4 --launcher pytorch --work-dir ./work_dirs/UltraSam
mim test mmdet configs/UltraSAM/UltraSAM_full/UltraSAM_point_refine.py --checkpoint ./work_dirs/UltraSam/iter_30000.pth
mim test mmdet configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py --checkpoint ./work_dirs/UltraSam/iter_30000.pth


mim train mmpretrain configs/UltraSAM/UltraSAM_full/downstream/classification/BUSBRA/resnet50.py \
    --work-dir ./work_dirs/classification/BUSBRA/resnet
mim train mmpretrain configs/UltraSAM/UltraSAM_full/downstream/classification/BUSBRA/MedSAM.py \
    --work-dir ./work_dirs/classification/BUSBRA/MedSam
mim train mmpretrain configs/UltraSAM/UltraSAM_full/downstream/classification/BUSBRA/SAM.py \
    --work-dir ./work_dirs/classification/BUSBRA/Sam
mim train mmpretrain configs/UltraSAM/UltraSAM_full/downstream/classification/BUSBRA/UltraSam.py \
    --work-dir ./work_dirs/classification/BUSBRA/UltraSam
mim train mmpretrain configs/UltraSAM/UltraSAM_full/downstream/classification/BUSBRA/ViT.py \
    --work-dir ./work_dirs/classification/BUSBRA/ViT

mim train mmdet configs/UltraSAM/UltraSAM_full/downstream/segmentation/BUSBRA/resnet.py \
    --work-dir ./work_dirs/segmentation/BUSBRA/resnet
mim train mmdet configs/UltraSAM/UltraSAM_full/downstream/segmentation/BUSBRA/UltraSam.py \
    --work-dir ./work_dirs/segmentation/BUSBRA/UltraSam_3000
mim train mmdet configs/UltraSAM/UltraSAM_full/downstream/segmentation/BUSBRA/SAM.py \
    --work-dir ./work_dirs/segmentation/BUSBRA/SAM
mim train mmdet configs/UltraSAM/UltraSAM_full/downstream/segmentation/BUSBRA/MedSAM.py \
    --work-dir ./work_dirs/segmentation/BUSBRA/MedSAM
```

</details>

## US-43d

Ultrasound imaging presents a substantial domain gap compared to other medical imaging modalities; building an ultrasound-specific foundation model therefore requires a specialized large-scale dataset. To build such a dataset, we crawled a multitude of platforms for ultrasound data. We arrived at US-43d, a collection of 43 datasets covering 20 different clinical applications, containing over 280,000 annotated segmentation masks from both 2D and 3D scans.

<details>
<summary>Click to expand datasets table</summary>

| Dataset               | Link                                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------------------------- |
| 105US                 | [researchgate](https://www.researchgate.net/publication/329586355_100_2D_US_Images_and_Tumor_Segmentation_Masks)   |
| AbdomenUS             | [kaggle](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm)                                               |
| ACOUSLIC              | [grand-challenge](https://acouslic-ai.grand-challenge.org/overview-and-goals/)                                     |
| ASUS                  | [onedrive](https://onedrive.live.com/?authkey=%21AMIrL6S1cSjlo1I&id=7230D4DEC6058018%2191725&cid=7230D4DEC6058018) |
| AUL                   | [zenodo](https://zenodo.org/records/7272660)                                                                       |
| brachial plexus       | [github](https://github.com/Regional-US/brachial_plexus)                                                           |
| BrEaST                | [cancer imaging archive](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/)                      |
| BUID                  | [qamebi](https://qamebi.com/breast-ultrasound-images-database/)                                                    |
| BUS_UC                | [mendeley](https://data.mendeley.com/datasets/3ksd7w7jkx/1)                                                        |
| BUS_UCML              | [mendeley](https://data.mendeley.com/datasets/7fvgj4jsp7/1)                                                        |
| BUS-BRA               | [github](https://github.com/wgomezf/BUS-BRA)                                                                       |
| BUS (Dataset B)       | [mmu](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)                                                          |
| BUSI                  | [HomePage](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)                                                      |
| CAMUS                 | [insa-lyon](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8g)        |
| CardiacUDC            | [kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset)                                    |
| CCAUI                 | [mendeley](https://data.mendeley.com/datasets/d4xt63mgjm/1)                                                        |
| DDTI                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                        |
| EchoCP                | [kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/echocp)                                                |
| EchoNet-Dynamic       | [github](https://github.com/echonet/dynamic)                                                                       |
| EchoNet-Pediatric     | [github](https://echonet.github.io/pediatric)                                                                      |
| FALLMUD               | [kalisteo](https://kalisteo.cea.fr/index.php/fallmud/#)                                                            |
| FASS                  | [mendeley](https://data.mendeley.com/datasets/4gcpm9dsc3/1)                                                        |
| Fast-U-Net            | [github](https://github.com/vahidashkani/Fast-U-Net)                                                               |
| FH-PS-AOP             | [zenodo](https://zenodo.org/records/10829116)                                                                      |
| GIST514-DB            | [github](https://github.com/howardchina/query2)                                                                    |
| HC                    | [grand-challenge](https://hc18.grand-challenge.org/)                                                               |
| kidneyUS              | [github](https://github.com/rsingla92/kidneyUS)                                                                    |
| LUSS_phantom          | [Leeds](https://archive.researchdata.leeds.ac.uk/1263/)                                                            |
| MicroSeg              | [zenodo](https://zenodo.org/records/10475293)                                                                      |
| MMOTU-2D              | [github](https://github.com/cv516Buaa/MMOTU_DS2Net)                                                                |
| MMOTU-3D              | [github](https://github.com/cv516Buaa/MMOTU_DS2Net)                                                                |
| MUP                   | [zenodo](https://zenodo.org/records/10475293)                                                                      |
| regPro                | [HomePage](https://muregpro.github.io/data.html)                                                                   |
| S1                    | [ncbi](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8205136/)                                                      |
| Segthy                | [TUM](https://www.cs.cit.tum.de/camp/publications/segthy-dataset/)                                                 |
| STMUS_NDA             | [mendeley](https://data.mendeley.com/datasets/3jykz7wz8d/1)                                                        |
| STU-Hospital          | [github](https://github.com/xbhlk/STU-Hospital)                                                                    |
| TG3K                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                        |
| Thyroid US Cineclip   | [standford](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5)                  |
| TN3K                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                        |
| TNSCUI                | [grand-challenge](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN-SCUI2020.md)        |
| UPBD                  | [HomePage](https://ubpd.worldwidetracing.com:9443/)                                                                |
| US nerve Segmentation | [kaggle](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data)                                              |


Once you downloaded the datasets:
Run each converter in ```datasets/datasets```

```bash
# run coco converters

# then preprocessing
python datasets/tools/merge_subdir_coco.py
python datasets/tools/split_coco.py
python datasets/tools/create_agnostic_coco.py path_to_datas_root --mode train
python datasets/tools/create_agnostic_coco.py path_to_datas_root --mode val
python datasets/tools/create_agnostic_coco.py path_to_datas_root --mode test
python datasets/tools/merge_agnostic_coco.py path_to_datas_root path_to_datas_root/train.agnostic.noSmall.coco.json --mode train
python datasets/tools/merge_agnostic_coco.py path_to_datas_root path_to_datas_root/val.agnostic.noSmall.coco.json --mode val
python datasets/tools/merge_agnostic_coco.py path_to_datas_root path_to_datas_root/test.agnostic.noSmall.coco.json --mode test
```

</details>

## References

If you find our work helpful for your research, please consider citing us using the following BibTeX entry:

```bibtex
@article{meyer2025ultrasam,
  title={Ultrasam: a foundation model for ultrasound using large open-access segmentation datasets},
  author={Meyer, Adrien and Murali, Aditya and Zarin, Farahdiba and Mutter, Didier and Padoy, Nicolas},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--10},
  year={2025},
  publisher={Springer}
}
```

---
