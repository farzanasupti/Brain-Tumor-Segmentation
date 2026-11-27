# TCGA-LGG (derived files)

`manifest.csv`, `manifest-slice-level.csv`, `folds.csv` and `folds-slice-level.csv`
are derived from the "Brain MRI segmentation" dataset by Mateusz Buda
(Kaggle `mateuszbuda/lgg-mri-segmentation`, licensed CC BY-NC-SA 4.0), whose
images come from The Cancer Imaging Archive (TCGA-LGG collection). These derived
files are shared under the same licence, CC BY-NC-SA 4.0.

Source paper: M. Buda, A. Saha, M. A. Mazurowski, "Association of genomic subtypes
of lower-grade gliomas with shape features automatically extracted by a deep
learning algorithm", Computers in Biology and Medicine 109 (2019) 218-225.

Images are not redistributed. Rebuild them with `prepare_external.py lgg`.
