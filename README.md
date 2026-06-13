# Comparing Machine Learning Methods for Algal Bloom Detection in Lough Neagh, Northern Ireland
This project aims to use machine learning methods, focusing on K-means clustering to 
### Project Motivation (Problem to be tackled)
https://sluggerotoole.com/2026/03/16/the-saga-of-lough-neaghs-deterioration-takes-another-toxic-turn/
https://user.eumetsat.int/resources/case-studies/loch-neagh-algal-bloom
## Notebook Layout
* **PART 1:** Fetching Data using Sentinel-2 + Indices Calculation
* **PART 2**: Algae Classification using Supervised Learning Methods
* **PART 3:** Image Rollout using Supervised Learning Methods
* **PART 4:** Unsupervised Learning (K-Means Clustering) **+ Results**
* **PART 5:** Explainable AI (SHAP) + Environmental Costs
## Introduction
### Geographical Context
Lough Neagh is situated in 
https://www.daera-ni.gov.uk/protected-areas/lough-neagh-assi
https://www.daera-ni.gov.uk/articles/action-being-taken-address-issues-lough-neagh
### Biological Context
https://isprs-archives.copernicus.org/articles/XLVIII-M-6-2025/251/2025/isprs-archives-XLVIII-M-6-2025-251-2025.pdf
### Previous Algal Bloom Events
https://www.niassembly.gov.uk/globalassets/documents/raise/topical-digests/td_lough-neagh_td06.pdf
https://www.belfasttelegraph.co.uk/news/environment/blue-green-algae-is-back-on-lough-neagh-after-warm-and-wet-weather/a/112203647.html
https://dataspace.copernicus.eu/gallery/2023-9-4-algae-bloom-lough-neagh
## Data Used and Indices Calculated
### Sentinel-2 Data
#### Sentinel-2 Infographic
https://sentinels.copernicus.eu/-/collection-1-level-2a
https://pmc.ncbi.nlm.nih.gov/articles/PMC7544481/
https://browser.stac.dataspace.copernicus.eu/collections/sentinel-2-l2a?.language=en
https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
### Indices Calculated
https://www.sciencedirect.com/science/article/pii/S2468312425000082
https://isprs-archives.copernicus.org/articles/XLVIII-M-6-2025/251/2025/isprs-archives-XLVIII-M-6-2025-251-2025.pdf
### IRIS Segmentation
## Unsupervised Machine Learning
### K-Means Clustering
## Supervised Machine Learning Methods
### Convolutional Neural Networks (CNN)
#### Introduction to CNNs

Convolutional Neural Networks, commonly known as CNNs, are a class of deep neural networks specially designed to process data with grid-like topology, such as images {cite}`Goodfellow-et-al-2016,lecun2015deep`. Originating from the visual cortex's biological processes, CNNs are revolutionising the way we understand and interpret visual data.

#### Why CNN for Image Data?

Traditional neural networks, when used for images, suffer from two main issues:

- **Too many parameters**: For a simple 256x256 colored image, an input layer would have \(256 * 256 * 3 = 196,608\) neurons, leading to an enormous number of parameters even in the first hidden layer.
- **Loss of spatial information**: Flattening an image into a vector for traditional neural networks can lose the spatial hierarchies and patterns in the image, which are often crucial for understanding and interpreting visual data.

CNNs address both issues by introducing convolutions.

#### Key Components of CNN

1. **Convolutional Layer** {cite}`lecun2015deep`: This is the core building block of a CNN. It slides a filter (smaller in size than the input data) over the input data (like an image) to produce a feature map or convolved feature. The primary purpose of a convolution is to extract features from the input data.
2. **Pooling Layer**: Pooling layers are used to reduce the dimensions of the feature maps, thereby reducing the number of parameters and computation in the network. The most common type of pooling is max pooling.
3. **Fully Connected Layer**: After several convolutional and pooling layers, the final classification is done using one or more fully connected layers. Neurons in a fully connected layer have connections to all activations in the previous layer, as seen in regular neural networks.
4. **Activation Functions**: Non-linearity is introduced into the CNN using activation functions. The Rectified Linear Unit (ReLU) is the most commonly used activation function in CNNs.

#### How CNNs Learn Spatial Hierarchies

CNNs learn spatial hierarchies automatically. The initial layers might learn to detect edges, the next layers learn to detect shapes by combining edges, further layers might detect more complex structures. This ability to learn spatial hierarchies from raw data gives CNNs their power. It allows them to detect complex objects in images by combining simpler features from the earlier layers.

#### Advantages of CNNs

- **Parameter Sharing**: A feature detector (filter) that's useful in one part of the image can be useful in another part of the image {cite}`krizhevsky2012imagenet`.
- **Sparsity of Connections**: In each layer, each output value depends only on a small number of input values, making the computation more efficient.

#### Basic Code Implementation

Below, you'll find a basic Convolutional Neural Network (CNN) structure implemented in TensorFlow. Treat this as a foundational blueprint for your subsequent implementations.
### Random Forests (RF)

Random Forest is a notable and significant part of machine learning and is commonly used for classification. It can also be used for regression, but its application in classification is more prevalent. Decision Trees are the core components of a Random Forest, so let's delve into the concepts of Decision Trees {cite}`breiman2001random,quinlan1986induction`.

#### Theoretical Foundations

#### 1. **Ensemble Learning**

Ensemble methods employ multiple learning algorithms to achieve better predictive performance than any individual learning algorithm alone {cite}`dietterich2000ensemble`. The primary principle behind ensemble models is that several weak learners come together to form a strong learner.

#### 2. **Decision Trees**

Decision trees are central to a Random Forest. They split data into subsets based on feature values, recursively producing a decision tree {cite}`quinlan1986induction`.

#### 3. **Bootstrap Aggregating (Bagging)**

Random Forests leverage bagging, where multiple dataset subsets are created by drawing samples with replacement. A separate decision tree is built for each of these samples {cite}`breiman1996bagging`.

#### 4. **Feature Randomness**

In conventional decision trees, the best feature is chosen to split data at every node. However, Random Forests introduce randomness by selecting a random set of features, then choosing the best split from this subset, ensuring a diverse ensemble of trees.


#### Advantages

- **Generalisation**: By combining the predictions of multiple trees, Random Forests tend to generalize better and are less susceptible to overfitting on training data.
  
- **Parallel Processing**: Each decision tree can be built independently, allowing for parallel processing which speeds up the algorithm considerably for large datasets.

- **Handling Missing Values**: Random Forests can handle missing values and still produce reasonable predictions.

- **Importance Scoring**: They provide an importance score for each feature, aiding in feature selection or interpretability.

#### Implementation in Python (Using Scikit-learn)
### Vision Transformers (ViT)

Vision Transformers (ViTs) are a recent breakthrough in the field of deep learning for image processing. They depart from the traditional convolutional neural network (CNN) approach and apply transformers, which were originally designed for natural language processing tasks, to image classification.

#### Theoretical Foundations

#### 1. **Tokenisation of Images**

Instead of processing images using convolutions, ViTs divide the image into fixed-size patches, linearly embed them, and then process the resulting sequence of vectors (or tokens) using a transformer.{cite}`dosovitskiy2020image`

#### 2. **Position Embeddings**

Since the original transformer doesn't have a notion of the relative positions of tokens, positional embeddings are added to the patch embeddings to retain the positional information.{cite}`dosovitskiy2020image`

#### 3. **Transformer Architecture**

The core of ViT is the transformer architecture, which consists of multiple layers of multi-head self-attention mechanisms and feed-forward neural networks.{cite}`dosovitskiy2020image`

#### 4. **Classification Head**

After processing through the transformer layers, the embedding of the first token (often referred to as the 'CLS' token) is used to classify the image.{cite}`dosovitskiy2020image`

#### Advantages of ViT

- **Model Transferability**: ViTs pre-trained on large datasets can be fine-tuned on smaller datasets, achieving high performance even when the available labeled data is limited.

- **Scalability**: ViTs are more data-hungry compared to CNNs. However, their performance continues to improve as the model size and the amount of data increase, often surpassing other architectures.

- **Flexibility**: The transformer architecture isn't specialized for grid-like data (like images), making ViTs potentially more flexible for varied input data types.

#### Challenges

- **Computational Demand**: ViTs can be computationally intensive, especially when dealing with large images or when the model has many layers.

- **Data Requirement**: To achieve optimal performance, ViTs often require more training data compared to CNNs.

#### Implementation
The implmentation of Vision Transformer is much more complicated than CNN and Random Forest as there is no built-in functions or layers in the library. However, the following code uses some existing functions like Muliti-head attention to build the transformer block. You don't need to know the exactly and detailed structure of ViT as it is not required in this course. Please follow the code below for example of implementation.
## Results
## Environmental Impact
Code carbon is used to quantify the energy used
### Algae Equivalent Metric
## Video Explanation of Code
## Infographic of Machine Learning Methods
This infographic displays the different machine learning methods implemented (Supervised and Unsupervised), as well as the Explainable AI implementation.
## References
European Space Agency (2023). S1 Products. [online] Copernicus.eu. Available at: https://sentiwiki.copernicus.eu/web/s1-products.
## Acknowledgements
This project was created as part of the final assignment for the module GEOL0069 AI for Earth Observations (AI4EO) at UCL, and incorprates python notebooks created by the module lead Michele Tasamados and teaching assistant Weibin Chen.
## Contact
Email: serenatrantcolab@gmail.com UCL Email: serena.trant.23@alumni.ucl.ac.uk
