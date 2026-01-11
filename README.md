# GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space

**Authors:** David G. Shatwell, Ishan Rajendrakumar Dave, Sirnam Swetha, Mubarak Shah  
**Affiliation:** Center for Research in Computer Vision, University of Central Florida (https://www.crcv.ucf.edu/)

**Paper:** https://arxiv.org/abs/2507.10473  
**Project page:** https://davidshatwell.com/gtloc.github.io/

This repository provides the official code and pretrained model for GT-Loc, a joint embedding framework that unifies geo-localization and time prediction from images. It includes evaluation scripts, dataset loaders, and configuration files to reproduce the paper's results across multiple benchmarks.

## Environment setup

Create and activate a conda environment (Python 3.10):
```bash
conda create -n gtloc python=3.10 -y
conda activate gtloc
```

Install required libraries:
```bash
pip install numpy pandas pillow tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Download pretrained model

```bash
pip install gdown
mkdir -p ckpts
gdown --id 170wGyCYLSF3DWOvITIAbtN1OkmDlS4v6 -O ckpts/gtloc.pt
```

## Download datasets

Follow the official instructions to obtain the datasets:

- **CVT & SkyFinder:** https://tsalem.github.io/DynamicMaps/
- **Im2GPS3K:** https://github.com/lugiavn/revisiting-im2gps?tab=readme-ov-file
- **GWS15k:** https://openaccess.thecvf.com/content/CVPR2023/papers/Clark_Where_We_Are_and_What_Were_Looking_At_Query_Based_CVPR_2023_paper.pdf

We cannot redistribute GWS15k images (Google Maps licensing), but we provide GPS coordinates. For access questions, please contact the dataset authors.

## Run evaluations

1. Update dataset paths in `src/configs/config.py`.
2. (If needed) recreate the GWS15k metadata CSV using `metadata/gws15k.csv` as a reference.
3. Run:
   ```bash
   bash ./eval.sh
   ```

We tested on a Quadro RTX 6000 (24GB). If you hit memory limits, reduce batch sizes in `src/configs/config.py` or adjust chunk sizes in `src/eval/eval_model.py`.

Results are printed to the terminal and saved to `output/eval_results.json`.

## Citation

If you use this work, please cite:
```bibtex
@InProceedings{Shatwell_2025_ICCV,
    author    = {Shatwell, David G. and Dave, Ishan Rajendrakumar and Swetha, Sirnam and Shah, Mubarak},
    title     = {GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {1-11}
}
```
