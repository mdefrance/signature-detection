# YOLOS for Handwritten Signature Detection

This repository hosts the code used to finetune YOLOS models to detect **handwritten signatures** in document images. The model is trained on the [`tech4humans/signature-detection`](https://huggingface.co/datasets/tech4humans/signature-detection) dataset, built from real-world signature data across diverse document types.

- Model: [`mdefrance/yolos-base-signature-detection`](https://huggingface.co/mdefrance/yolos-base-signature-detection)
- Based on: [`hustvl/yolos-base`](https://huggingface.co/hustvl/yolos-base)
- Dataset: [`tech4humans/signature-detection`](https://huggingface.co/datasets/tech4humans/signature-detection)
- License: Apache 2.0

---

## Quick Usage

```python
from datasets import load_dataset
from transformers import pipeline

# Load test image from dataset
dataset = load_dataset("samuellimabraz/signature-detection")
image = dataset["test"][0]["image"]

# Load the finetuned pipeline
yolos = pipeline(
    task="object-detection",
    model="mdefrance/yolos-base-signature-detection",
    device_map="auto"
)

# Run inference
prediction = yolos(image)

```


## Performances


| **Metric**                      | [yolos-base-signature-detection](https://huggingface.co/mdefrance/yolos-base-signature-detection) | [yolos-base-signature-detection](https://huggingface.co/mdefrance/yolos-base-signature-detection) | [yolos-base-signature-detection](https://huggingface.co/mdefrance/yolos-base-signature-detection) | 
|:--------------------------------|------------:|-----------:|-----------------------------:|
| **Inference Time - CPU (s)**    |    2.250    |      0.787 |                   **0.262**  |
| **Inference Time - GPU (s)**    |     1.464   |      0.023 |                   **0.014**  |
| **Parameters**                  |   127.73M   |     30.65M |                        6.47M |
| **mAP50**                       |   **0.887** |      0.859 |                        0.856 |
| **mAP50-95**                    |   **0.495** |      0.421 |                        0.395 |

Inference times are computed on a laptop with following specs:
* CPU: Intel Core i7-9750H
* GPU: NVIDIA GeForce GTX 1650


## Setup & Usage (Local)

### Requirements

- Python **3.12.1** (not tested for other versions)
- [Poetry](https://python-poetry.org/) **2.1.3** for dependency management

### Installation

1. Clone the repo:

```bash
git clone https://github.com/mdefrance/signature-detection.git
cd signature-detection
```


2. Install dependencies with Poetry:

```bash
pip install poetry==2.1.3
poetry install
```

### Fine-Tuning Notebook

The notebook used to fine-tune YOLOS on the signature detection dataset is available [here](https://github.com/mdefrance/signature-detection/blob/main/src/fine-tuning.ipynb). It includes:

- Dataset loading & preprocessing
- YOLOS training configuration
- Training with Pytorch Lightning Trainer
- Evaluation & visualization of predictions


## Training Data

- Training: 1,980 images (70%)
- Validation: 420 images (15%)
- Testing: 419 images (15%)
- Format: COCO JSON
- Resolution: 640x640 pixels
- Sources: [Tobacco800](https://paperswithcode.com/dataset/tobacco-800) + [signatures-xc8up](https://universe.roboflow.com/roboflow-100/signatures-xc8up)
- Preprocessed by [Samuel Lima Braz](https://huggingface.co/samuellimabraz) via [Roboflow](https://roboflow.com/)

## Out-of-Scope Use
- Forgery or Fraudulent Use: This model is for detection only, not signature generation or spoofing.
- Non-Signature Detection: Not suitable for detecting other objects or text.
- High-Stakes Automation: Use human verification in legal or financial applications.

## Limitations & Ethical Use

- Bias Risk: Limited performance on unseen signature styles or document types.
- False Positives/Negatives: Manual review is recommended in production.
- Image Quality: Model struggles with noisy, low-res, or poorly lit inputs.
- Ethics: Respect privacy and regulatory compliance when deploying.

## Citation

If you use this model, consider citing the original YOLOS paper:


**BibTeX:**
```bibtex
@article{DBLP:journals/corr/abs-2106-00666,
  author    = {Yuxin Fang and
               Bencheng Liao and
               Xinggang Wang and
               Jiemin Fang and
               Jiyang Qi and
               Rui Wu and
               Jianwei Niu and
               Wenyu Liu},
  title     = {You Only Look at One Sequence: Rethinking Transformer in Vision through
               Object Detection},
  journal   = {CoRR},
  volume    = {abs/2106.00666},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.00666},
  eprinttype = {arXiv},
  eprint    = {2106.00666},
  timestamp = {Fri, 29 Apr 2022 19:49:16 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-00666.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### Additional Resources 

- **Blog post of comparison of Signature Detection Models:** [Hugging Face Blog](https://huggingface.co/blog/samuellimabraz/signature-detection-model)
- **Blog post associated Finetuning Notebook:** [Google Colab Notebook](https://colab.research.google.com/drive/1wSySw_zwyuv6XSaGmkngI4dwbj-hR4ix)
- **Finetuning of YOLOS Notebook Example:** [Google Colab Notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/Fine_tuning_YOLOS_for_object_detection_on_custom_dataset_(balloon).ipynb)


## **Author**

<div align="center">
  <table>
    <tr>
      <td align="center" width="160">
        <a href="https://huggingface.co/mdefrance">
          <img src="https://avatars.githubusercontent.com/u/74489838?v=4" width="160" alt="Mario DEFRANCE"/>
          <h3>Mario DEFRANCE</h3>
        </a>
        <p><i>Data Scientist / AI Engineer</i></p>
      </td>
      <td width="500">
        <h4>Responsibilities in this Project</h4>
        <ul>
          <li>🔬 Model development and training</li>
          <li>⚙️ Performance evaluation</li>
          <li>📝 Technical documentation and model card</li>
        </ul>
      </td>
    </tr>
  </table>
</div>
