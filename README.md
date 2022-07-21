<p align="center">
    <h1 align="center">
        <p>Handwritten Character Recognition - an unofficial implementation of the paper</p>
        <a href="https://arxiv.org/abs/2109.10282">TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models</a>
    </h1>
    
</p>

---

This is an unofficial implementation of TrOCR based on the [Hugging Face transformers library](https://huggingface.co/docs/transformers/model_doc/trocr) and the [TrOCR paper](https://arxiv.org/abs/2109.10282). There is also a repository by the authors of the paper ([link](https://github.com/microsoft/unilm/tree/master/trocr)). The code in this repository is merely a more simple wrapper to quickly get started with training and deploying this model for character recognition tasks.

&nbsp;

## Results:

![Predictions](.resources/example-predictions.png)

After training on a dataset of 2000 samples for 8 epochs, we got an accuracy of 96,5%. Both the training and the validation datasets were not completely clean. Otherwise, even higher accuracies would have been possible.



&nbsp;

## Architecture:

![TrOCR](.resources/architecture.jpg)
(TrOCR architecture. Taken from the original paper.)

[TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282), Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei, Preprint 2021.

&nbsp;

&nbsp;

&nbsp;

---
---
&nbsp;



# 1. Setup

Clone the repository and make sure to have conda or miniconda installed.
Then go into the directory of the cloned repository and run

``` bash
conda env create -n trocr --file environment.yml
conda activate trocr
```

This should install all necessary libraries.


### Training without GPU:

It is highly recommended to use a CUDA GPU, but everything also works on cpu. For that, install from file `environment-cpu.yml` instead.

In case the process terminates with the warning "killed", reduce the batch size to fit into the working memory.

&nbsp; 

# 2. Using the repository

There are 3 modes, inference, validation and training. All 3 of them can either start with a local model in the right path (see `src/constants/paths`) or with the pretrained model from huggingface. Inference and Validation use the local model per default, training starts with the huggingface model per default.

&nbsp; 

### Inference (Prediction):

``` bash
python -m src predict <image_files>  # predict image files using the trained local model
python -m src predict data/img1.png data/img2.png  # list all image files
python -m src predict data/*  # also works with shell expansion
python -m src predict data/* --no-local-model  # uses the pretrained huggingface model
```

### Validation:

``` bash
python -m src validate # uses pretrained local model
python -m src validate --no-local-model # loads pretrained model from huggingface
```

### Training:

```bash
python -m src train  # starts with pretrained model from huggingface
python -m src train --local-model  # starts with pretrained local model
```

&nbsp; 

For validation and training, input images should be in directories `train` and `val` and the labels should be in `gt/labels.csv`.
In the csv, each row should consist of image name and then ending, for example `img1.png,a` (in quotes, if necessary).

It is also pretty straightforward to read labels from somewhere else. For that, just add the necessary code to `load_filepaths_and_labels` in `src/dataset.py`.

For choosing a subsample of the train data as validation data, this command can be used

``` bash
find train -type f | shuf -n <num of val samples> | xargs -I '{}' mv {} val
```

&nbsp; 

# 3. Integrating into other projects

If you want to use the predictions as part of a bigger project, you can just use the interface provided by the `TrocrPredictor` in main. For that make sure to run all code as [python modules](https://docs.python.org/3/tutorial/modules.html).


See the following code example:

``` python
from PIL import Image
from trocr.src.main import TrocrPredictor

# load images
image_names = ["data/img1.png", "data/img2.png"]
images = [Image.open(img_name) for img_name in image_names]

# directly predict on Pillow Images or on file names
model = TrocrPredictor()
predictions = model.predict_images(images)
predictions = model.predict_for_file_names(image_names)

# print results
for i, file_name in enumerate(image_names):
    print(f'Prediction for {file_name}: {predictions[i]}')
```

&nbsp;

# 4. Adapting the Code

In general, it should be easy to adapt the code for other input formats or use cases.

- Learning Rate, Batch size, Train Epoch Count, Logging, Word Len: `src/configs/constants.py`
- Input Paths, Model Checkpoint Path: `src/configs/paths.py`
- Different label format: `src/dataset.py : load_filepaths_and_labels`

The word len constant is very important. To facilitate batch training, all labels need to be padded to the same length.
Some experiments might be needed here. For us, padding to 8 worked well.

If you want to change specifics of the model, you can supply a TrOCRConfig object to the transformers interface.
See <https://huggingface.co/docs/transformers/model_doc/trocr#transformers.TrOCRConfig> for more details.

&nbsp;

# 5. Contact

If the setup fails to work, please let me know in a Github issue! Sometimes sub-dependencies update and become incompatible with other dependencies, so the dependency list has to be updated.

Feel free to submit issues with questions about the implementation as well.

For questions about the paper or the architecture, please get in touch with the authors.
