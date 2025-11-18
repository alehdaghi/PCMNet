# PCMNet — Part-Prototypical Concept Mining Network

**PCMNet** (Beyond Patches) is an ante-hoc, part-based prototype learning framework that discovers adaptive, semantically coherent regions and clusters them into concept centroids for interpretable classification and explanation. PCMNet learns *part-prototypes* (localized regions) and aggregates them into *concepts* (groups of related part-prototypes) via the Concept Mining Module (CMM). The model produces sparse Concept Activation Vectors (CAVs) used for classification and explanation.

Paper (AAAI 2026): *Beyond Patches: Mining Interpretable Part-Prototypes for Explainable AI*  
Project / code: https://github.com/alehdaghi/PCMNet


| ![Intro](https://github.com/alehdaghi/PCMNet/blob/main/imgs/intro-c.jpg) | 
|:--:| 
| *Explanation methods. (a) Heatmap-based methods see the feature backbones as black-box models, and try to locate regions activated by the most important features. (b) Patch-based methods decompose the input into image patches and explain the decision using patch-derived components. (c) PCMNet, in contrast, mines both prototypical and non-prototypical concepts, offering a broader and more interpretable set of regions to explain the model decision.* |

---

## Features
- Unsupervised part discovery (learned masks) instead of fixed patches
- Marginal Cluster Center (MCC) loss to promote semantically coherent parts
- Two-level clustering (part → concept) to build concept centroids (CMM)
- Sparse classifier over CAVs for interpretable decisions
- Metrics & tools for faithfulness, stability, consistency, sparsity, occlusion robustness
- Lightweight overhead: +~0.4M params, +~0.3G FLOPs (ResNet50 baseline)


<table style="width:100%; text-align:center;">
  <tr>
    <td align="center"> <img src="https://github.com/alehdaghi/PCMNet/blob/main/imgs/model-c.png" style="width:100%; max-width:550px;"> (a) </td>
    <td align="center" ><img src="https://github.com/alehdaghi/PCMNet/blob/main/imgs/CMM-c.png" style="width:100%; max-width:550px;">(b)></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:left; padding:10px;">
      (a) <i> Overall architecture of PCMNet: part prototypes are learned from spatial features, then clustered within each class to form concept prototypes. Distances to these concept clusters yield activated concepts for interpretable classification.  </i>
      <br><br>
      (b) <i> Concept Mining Module: DBSCAN is applied within each class to form part-centroid clusters. Their centers serve as concept prototypes, and similarity (inverse distance) between part features and these centroids forms the CAVs used in the final decision. </i>
    </td>
  </tr>
</table>
---
