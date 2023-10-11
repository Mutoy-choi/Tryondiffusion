# Tryondiffusion

## Introduction
This repository is an attempt to implement the Tryondiffusion model. For more details, visit the [official Tryondiffusion website](https://tryondiffusion.github.io/).

![image](https://github.com/Mutoy-choi/Tryondiffusion/assets/87027571/a3b9b53c-f6a3-4a52-8c3d-e7c26d50c55e)

## Environment

The code was developed and tested on the following environment:

* Operating System: Windows server 2019

* Python Version: Python 3.10

* GPU : NVIDIA Tesla T4 16GB

* CUDA 11.8



## Getting Started

### Clone the Repository
To get started with training examples, first clone this repository by running the following command in your terminal:

```bash
git clone https://github.com/Mutoy-choi/Tryondiffusion


cd Tryondiffusion
```
This will clone the repository and navigate you into the project directory.

### Set Up Virtual Environment

```
python -m venv venv

.\venv\Scripts\activate
```

These commands create and activate a virtual environment named venv. This isolates the project dependencies, making it easier to manage.

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Example Training code

```
python one_shot_test_ParallelUnet.py
```

This file allows you to know if the model is working well or not using example data for ParallelUnet(From 128x128 to 256*256)

## Further upload
* update preprocessing AI-Hub data
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=78
  
 Deployment content and amount of data provided

Studio fashion video (model photo) 6,741,328 cases

Studio Fashion Video Model Key Points: 120,936 Cases

Studio Fashion Video Model Semantic: 120,936 cases

Fashion products and fashion video pair: 117,270 cases

Fashion product representative photos (product photos) 40,036 cases

Fashion product key points: 40,036 cases

Fashion Product Semantic: 40,036 cases
  
