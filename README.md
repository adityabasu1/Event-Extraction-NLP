# Joint Event-Relation Extraction using Encoder Decoder Architecture

Natural Language Processing Course Project of Group 19 at IIT Kharagpur (Autumn'21) <br />

Group Members: <br />
- Abhikhya Tripathy (19EC10085)
- Debaditya Mukhopadhyay (19IE10036)
- Aditya Basu (19IE10002)
- Angana Mondal (19IE10039)


## Introduction <br />
Joint-event-extraction is a significant emerging application of NLP techniques which involves extracting structural information (i.e., event triggers, arguments of the event) from unstructured real-world corpora. Existing methods for this task rely on complicated pipelines prone to error propagation. 

## Model Architecture <br />
An encoder-decoder based architecture for joint entity-relation extraction was proposed by [Tapas Nayak et al.](https://arxiv.org/pdf/1911.09886.pdf), and we further develop the architecture to deploy it to predicting ``` trigger```, ```argument``` and ```relation``` tuple (including the ```classes of the trigger and argument```). We also utilise pretrained BERT embeddings to preprocess our data.  <br />
<br />

<p align="center">
  <img width="300" src="https://cdn.discordapp.com/attachments/793393040375021591/913798876677410826/unknown.png" alt="Model Architecture">
</p>

## Datasets <br />
The data is available at: https://drive.google.com/drive/u/1/folders/1fYP9PUQYRV0JWBa-N3CwuGkOCeBielT9 <br />
To obtain the Word2Vec embeddings and BERT embeddings, download ```'w2v.txt'``` and ```'BERT_embeddings.txt'``` from the aforementioned link. <br />

## Requirements <br />
- Python 3.5 +
- Pytorch 1.1.0
- CUDA 8.0

## Running the Code <br />
- **Python3:** ```python3  Joint_Event_Extraction.py  gpu_id  random_seed  source_data_dir  target_data_dir  train/test  w2v/bert```

- **IPython:** Run individual cells of ```NLP_Proj6_Grp_19.ipynb```
    - set the data folder and the output saving folder:
  ```src_data_folder = path```
  ```trg_data_folder = path + 'Model'```
  
    - for switching between training and testing phases, set the following parameters under ```if __name__ == "__main__":```,
  ```job_mode = 'train'``` or ```job_mode = 'test'```

    - for switching between Word2Vec and BERT embeddings, set the following parameters under ```if __name__ == "__main__":```,
  ```embedding_type = 'w2v'``` or ```embedding_type = 'bert'```

**Command Line Arguments**
- ```Source_data_dir```: Path to source data directory
- ```Target_data_dir```: Path to target data directory
- ```train/test```: Job mode (Choose ```only one``` of the two modes at once)
- ```w2v/bert```: Embedding type

**Default Command Line Arguments for Google Colaboratory**
- os.environ[‘CUDA_VISIBLE_DEVICES’] (= ```gpu_id```) = ‘0’
- ```random_seed``` = 42

