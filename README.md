# A deep-learning multitask approach for staples and wound segmentation in abdominal post-surgical images

## Authors: Gabriel Moyà-Alcover, Miquel Miró-Nicolau, Marc Munar and Manuel González-Hidalgo

## Abstract

Deep learning techniques provide a powerful and versatile tool in different areas, such as object segmentation in medical images. In this paper, we propose a network based on the U-Net architecture to perform the segmentation of wounds and staples in abdominal surgery images. Moreover, since both tasks are highly interdependent, we propose a multi-task architecture that allows to simultaneously obtain, in the same network evaluation, the masks with the staple location and the wound of the image. When performing this multitasking, it is necessary to formulate a global loss function that linearly combines the losses of both partial tasks. This is why the study also involves the GradNorm algorithm to determine which weight is associated with which loss function. The main conclusion of the study is that multi-task segmentation offers superior performance compared to segmenting different methods separately, and similar performance to that provided by GradNorm.

### Reference

```bibtex
@inproceedings{moya2023multitask,
  title={A multitask deep learning approach for staples and wound segmentation in abdominal post-surgical images},
  author={Moy{\`a}-Alcover, Gabriel and Mir{\'o}-Nicolau, Miquel and Munar, Marc and Gonz{\'a}lez-Hidalgo, Manuel},
  booktitle={Conference of the European Society for Fuzzy Logic and Technology},
  pages={208--219},
  year={2023},
  organization={Springer}
}
````
