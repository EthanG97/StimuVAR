# StimuVAR
The official implementation of [_StimuVAR: Spatiotemporal Stimuli-aware Video Affective Reasoning with Multimodal Large Language Models_](https://arxiv.org/abs/2409.00304) accepted by *IJCV*.



## Install

1. Clone this repository and navigate to StimuVAR folder

```bash
git clone https://github.com/EthanG97/StimuVAR.git
cd StimuVAR
```
2. Install Packages
```bash
conda create -n stimuvar python=3.9 -y
conda activate stimuvar
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

In this work, we conduct experiments on four datasets: [_VCE_](https://drive.google.com/drive/folders/1sRKitbXpLZ4pwXTONjiA-X0Y1z4I2o4X), [_VE-8_](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA), [_YF-6_](https://drive.google.com/drive/folders/11uJTbqdHXqjQw63_teXXda0cILGDVyo4), and [_EmoSet_](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?rlkey=7f3oyjkr6zyndf0gau7t140rv&e=2&dl=0). 

The following examples use the VCE dataset for demonstration purposes.




## 🎞️ Event-driven Frame Sampling

We provide a preprocessing script to extract motion-salient frames from video clips using dense optical flow analysis. This is particularly useful when preparing frame-level stimuli that capture rapid, key events—often corresponding to dramatic changes in a video's visual dynamics.

```bash
python helpers/extract_frames.py \
  --input_json helpers/alltrain.json \
  --video_root /path/to/videos \
  --output_dir /path/to/output_frames \
  --total_frames 6
```


## Training StimuVAR

The training pipeline for **StimuVAR** consists of two sequential stages:

- **Stage 1**: Visual feature alignment  
- **Stage 2**: Emotion reasoning based on aligned features

You can download the necessary resources from Google Drive:

- [📁 Training Data](https://drive.google.com/file/d/1-M8rQcxNmTKL6rahTq1g8QDT0shyn2H0/view?usp=drive_link)  
- [📁 Test Data](https://drive.google.com/file/d/1G_WjRd3FPkFexxSOii_t3vnc4i20yW_z/view?usp=drive_link)  
- [📄 Reasoning File](https://drive.google.com/file/d/1V6E68KOCO3qiUacg_v_XCMbdsYhl4dTu/view?usp=drive_link)
- [Trained model](https://drive.google.com/drive/folders/1hsX2YSVSG1GkCXpGwzRRwEIfTu9RmANP?usp=drive_link)

### Training Commands

```bash
# Stage 1: Train for visual feature alignment
torchrun --nproc_per_node=1 train.py --config config/Stage1.yaml

# Stage 2: Train for emotion reasoning
# (Use the Stage 1 model as the base model, specified in the Stage2 config)
torchrun --nproc_per_node=1 train.py --config config/Stage2.yaml
```

## Inference

To run inference with a trained StimuVAR model:

1. **Update the configuration**  
   Open `config/Inference.yaml` and set the `model_path` to the checkpoint of your trained Stage 2 model.

2. **Run the inference script**  
   Use the following command:

   ```bash
   torchrun --nproc_per_node=1 inference.py --config config/Inference.yaml
   ```

## Demo
  To run the demo for a single video, fill in the model path, config file and the video name
  ```bash
python demo.py \
  --video assets/sample_video.mp4 \
  --model_path checkpoints/stage2/checkpoint-150000 \
  --config config/Inference.yaml
```
## 🧪 Evaluation

### CLIP-Score


```bash
python Metrics/Clip_score/clip_score.py \
  --response_path /path/to/model_responses.jsonl \
  --img_dir /path/to/extracted_test_set_frames/
python Metrics/Clip_score/ave.py \
  --filename /path/to/_clipscore.json
```

### LLM-judge
```bash
python Metrics/LLM_judge/LLMjudge.py </path/to/model_responses.jsonl> <output_file>
```

### Doubly-Right & Rank3
```bash
python Metrics/Doubly/gpt_predict.py --response_path /path/to/model_responses.jsonl  
#Emotion prediction based on reason using GPT
python Metrics/Doubly/emo_align.py --response_path /path/to/model_responses.jsonl

```


### 🙏 Acknowledgment

Special thanks to [_Valley_](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?rlkey=7f3oyjkr6zyndf0gau7t140rv&e=2&dl=0) for providing high-quality code, which served as the foundation for our implementation.


### 📖 Citation

If you find this project helpful in your research, please consider citing our paper:

```bibtex
@article{guo2025stimuvar,
  title     = {StimuVAR: Spatiotemporal Stimuli-aware Video Affective Reasoning with Multimodal Large Language Models},
  author    = {Guo, Yuxiang and Siddiqui, Faizan and Zhao, Yang and Chellappa, Rama and Lo, Shao-Yuan},
  journal   = {International Journal of Computer Vision},
  pages     = {1--17},
  year      = {2025},
  publisher = {Springer}
}
