# **UltraSam: A Foundation Model for Ultrasound using Large Open-Access Segmentation Datasets**

_Adrien Meyer, Aditya Murali, Didier Mutter, Nicolas Padoy_

[![arXiv](https://img.shields.io/badge/arxiv-2307.15220-red)](https://arxiv.org/pdf/2411.16222)

![UltraSam](./assets/UltraSam_main.png)

## What's New

**06/01/2025**: Added a detailed table with links to individual datasets that constitute our compiled US-43d.


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

</details>

🚨  **Disclaimer** :

The code repository is currently private. The full code will be made publicly available upon acceptance of the associated publication. For now, only the pre-trained model checkpoint is accessible [at this link](https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth).

Stay tuned for updates!

## References

If you find our work helpful for your research, please consider citing us using the following BibTeX entry:

```bibtex
@article{meyer2024ultrasam,
  title={UltraSam: A Foundation Model for Ultrasound using Large Open-Access Segmentation Datasets},
  author={Meyer, Adrien and Murali, Aditya and Mutter, Didier and Padoy, Nicolas},
  journal={arXiv preprint arXiv:2411.16222},
  year={2024}
}
```

---
