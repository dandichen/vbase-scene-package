# Class `Classifier`
Image classification module base class, provides a uniform interface for different classification algorithm.
## Variables
## Methods
### `__init__` 
pass
### `predict`
Predict tag confidence with a list of numpy arrays as input
**Param:**
-	`in_list_numpy` : list of numpy array, containing input images
**Return:**
- `label`: a `n`*`m` numpy array indicating confidence of each of `m` tags for each one of the `n` input frames.
----------------------------

# Class `TradClassifier`(`Classifier`)
 A wrapper for classical feature-classifier scene classification system
## Variables
- `feature_extractor`: a `FeatureExtractor` object. Contained feature extraction module
- `predictor`: a `Predictor` object or a list of `Predictor` objects. Contained classification algorithm module

## Methods
### `__init__`
Initialize `feature_extractor` and `predictor` 

**Param:**
- `config`: Config parser pointing to config file of scene classification
- `config_label`: Corresponding label in config file
----------------------------

# Class `End2EndClassifier`(`Classifier`)
This class is the wrapper for deep learning end to end classification system. 
## Variables
- `model_params`: The end to end model parameters being used in current prediction, e.g. model iterations, contexts, output layer.
- `batch_size` : number of instances in a batch.
- `mean_file`: path to mean image as .nd
- `mean_img`: mean image as numpy array
- `classifier`: mxnet classifier object
- `h`,`w`: height and width of input images
## Methods
### `__init__`
Preload the deep classification model.
**Param:**
- `config`: Config parser pointing to config file of deep learning classifier
- `config_label`: Corresponding label in config file

### `predict`
Predict tag confidence with a list of numpy arrays or a list of image path as input
**Param:**
-	`in_list_numpy`or`in_list_path` : list of numpy array, containing input images/list of string, indicating the file path of input images. Only one of the input can be used a time.
**Return:**
- `label`: a n*m numpy array indicating confidence of each of m tags for each one of the n input frames.
----------------------------

# Class `FeatureExtractor`
Image feature extraction module base class.
## Variable

## Method
----------------------------

# Class `Predictor`
General classifier module base class. 
## Variable
## Method
### `__init__`
### `predict`
Get predicted label for each input instance.
### `predict_prob`
Get predicted probability for each input instance for each class.

----------------------------

# Class `TrainingUtil`
A convenient training and validation tool for frame-wise scene classifier
## Variable
- `learner`
- `feature_extractor`
- `input_data`
- `output_dir`
## Method
### `__init__`
### `learn`
----------------------------
# Class `Learner`
General classifier learning process base class.
## Variable
## Method
----------------------------
# Class `Pipeline`
General video processing pipeline. Take a video as input, calculating every clip with certain semantic meaning in the video. 
## Variables
- `classifier`: a `Classifier` object or a list of `Classifier` object. Scene classification module used in such pipeline.
- `aggregator`: a `Aggregator` object. Aggregation module used in such pipeline. 
## Methods
###  `__init__`

**Param:**
- `config`: RawConfigParser object point to the config file of pipeline.
- `config_label`: The section label in config file which the correct config corresponds to.
### `get_result`
Take a video as input, output starting and ending point for clip of certain tag in the form of seconds.
**Param:**
- `in_video`: a string indicating the path to the input video.

**Return:**
- `result`: a list of lists, each lists contains tuples indicating starting and ending points for each clip of a certain semantic tag, in the form of seconds.
### `get_result_sec`
Take a video as input, output starting and ending point for clips of certain tag in the form of number of frames.
**Param:**
- `in_video`: a string indicating the path to the input video.

**Return:**
- `result`: a list of lists, each lists contains tuples indicating starting and ending points for each clip of a certain semantic tag, in the form of frame number.

----------------------------

# Class `Aggregator`
Aggregation process base class. Take prediction confidence for multiple tags as input, generating clips as output.
## Variables
## Methods
### `__init__`

### `get_result`
Take a series of prediction result as input, output aggregated confidence score for each frame for visualization and validation.
**Param:**
- `in_pred`: a `n`*`m` numpy array indicating predicted confidence of `m` tags for each of the `n` frame.
- `in_video`: a string indicating path to input video.

**Result:**
- `result_confidence`: a `t`*`m` numpy array indicating aggregated confidence of `m` tags for each of the `t` frame.
### `get_result_clip`
Take a series of prediction result as input, output starting and ending point for clips for every tag. **Param:**
- `in_pred`: a `n`*`m` numpy array indicating predicted confidence of `m` tags for each of the `n` frame.
- `in_video`: a string indicating path to input video.
- `in_threshold`: a size `m` numpy array indicating confidence threshold for each tag to consider as positive. Default 0.5 for every tag.

**Result:**
- `result`: a list of `m` lists, each lists contains tuples indicating starting and ending points for each clip of a certain semantic tag, in the form of frame number.
