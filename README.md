
<h2 align="center"> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/34601">SimRP: Syntactic and Semantic Similarity Retrieval Prompting Enhances Aspect Sentiment Quad Prediction</a></h2>
<h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2>

<h4 align="center">

🚀 Welcome to the repo of [**SimRP**](https://github.com/jian-projects/simrp)!

CCM addresses the sentiment quad extraction task by retrieving syntactic similar samples, accepted by AAAI2025.

<!-- [![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-Uni_MoE-yellow)](https://huggingface.co/Uni-MoE) -->
<!-- [![Project Page](https://img.shields.io/badge/Project_Page-Uni_MoE-blue)](https://uni-moe.github.io/) -->
<!-- [![Demo](https://img.shields.io/badge/Demo-Local-orange)](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master?tab=readme-ov-file#-demo-video)  -->
<!-- [![Paper](https://img.shields.io/badge/Paper-arxiv-yellow)](https://arxiv.org/abs/2405.11273) -->

[Zhongquan Jian](https://scholar.google.com/citations?user=C1PWVBUAAAAJ&hl=zh-CN), Yanhao Chen, Jiajian Li, Shaopan Wang, Xiangjian Zeng, [Junfeng Yao](https://scholar.google.com/citations?hl=zh-CN&user=Szz3hSMAAAAJ), Xinyin An, [Qingqiang Wu](https://dblp.uni-trier.de/pid/130/0742.html)
</h4>

<!-- ## 🌟 Structure

The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/model.png" height="100%" width="75%"/></div> -->

## ⚡️ Install

The following instructions are for Linux installation.
We would like to recommend the requirements as follows.
* Python == 3.9.16
* CUDA Version >= 11.7

1. Clone this repository and navigate to the ccm folder
```bash
git git@github.com:jian-projects/simrp.git
cd simrp
```

2. Install Package
```Shell
conda create -n simrp python==3.9.16
conda activate simrp
# pip install -r env.txt
```

<!-- 3. Replace all the absolute pathnames '/path/to/' with your specific path to the Uni-MoE file
**(Including all the eval_x.py/inference_x.py/train_mem_x.py/data.py/demo.py files and config.json files from the model weights)** -->


## 🌈 How to train and inference

1. Make sure that all the weights are downloaded and the running environment is set correctly.

2. run the script:
```bash
python run_simrp.py
```

<!-- 2. run inference scripts [`inference_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_audio.sh) and [`inference_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_speech.sh) using ```bash inference_audio.sh``` ```bash inference_speech.sh``` or run the following commands to inference:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_audio/inference_all.py
```
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_speech/inference_all.py
```
To launch the online demo ( It is highly recommended to launch the demo with [Uni-MoE-speech-v1.5](https://huggingface.co/VictorJsy/Uni-MoE-speech-v1.5) that need the basic parameters of [Uni-MoE-speech-base-interval](https://huggingface.co/VictorJsy/Uni-MoE-speech-base-interval)), run:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python demo/demo.py
python demo/app.py
``` -->

## Citation

If you find SimRP useful for your research and applications, please cite using this BibTeX:
```bibtex

@INPROCEEDINGS{2025.absa.asqp.simrp,
    title={SimRP: Syntactic and Semantic Similarity Retrieval Prompting Enhances Aspect Sentiment Quad Prediction}, 
    author={Jian, Zhongquan and Chen, Yanhao and Li, Jiajian and Wang, Shaopan and Zeng, Xiangjian and Yao, Junfeng and An, Xinying and Wu, Qingqiang}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={39}, 
    number={23}, 
    year={2025}, 
    pages={24248-24256},
    doi={https://doi.org/10.1609/aaai.v39i23.34601}
}
```