# MetaCortex  
**One Cortex, Many Modalities**  

*Unifying Text, Image, and Tabular Data Modeling for End-to-End ML Workflows*

**Integrated Pipelines for Sequential Text Generation, Pixel-Based Image Classification,  
and Classical Classifier Benchmarking**

MetaCortex is a series of Python-based machine learning and deep learning implementations  
that demonstrate hands-on skills in data processing, model development, and evaluation.  
This project emphasizes technical rigor in data preparation, model design, performance  
benchmarking, and predictive analysis.

___

## Models

### Stacked LSTM Character Sequence Generator

Character-level sequential text generation based on stacked Long Short-Term Memory (LSTM)  
layers trained on literary datasets.

**Features:**
- Text preprocessing: normalization, tokenization, and stopword filtering  
- Sequence encoding: character-to-index transformation of input data  
- Model architecture: multi-layer LSTM network with dropout regularization  
- Checkpointing: saving and restoring optimal model weights during training  
- Text generation: generation of extended sequences based on learned patterns  

___

### Random Forest Pixel-Based Digit Classifier

Image classification pipeline identifying handwritten digits from grayscale pixel data  
using Random Forest ensemble methods.

**Features:**
- Data parsing: loading pixel intensity values from CSV format  
- Visualization: rendering digit images for exploration and verification  
- Dataset preparation: partitioning into training and test sets  
- Model training: Random Forest with 100 decision trees  
- Evaluation: accuracy scoring based on prediction correctness  

___

### Stratified Cross-Validated Classifier Benchmark

Comparative evaluation of classical machine learning algorithms on structured  
tabular data.

**Features:**
- Exploratory data analysis: statistical summaries, histograms, boxplots, and scatter matrices  
- Train-test splitting: stratified partitioning to preserve class distribution  
- Cross-validation: applied to Logistic Regression, LDA, KNN, Naive Bayes, and SVM  
- Performance aggregation: mean accuracy and standard deviation across folds  
- Visualization: boxplots comparing algorithm performance  
- Final model evaluation: accuracy score, confusion matrix, and classification report for SVM  

___

## Tech Stack

- **Language:** Python 3.x  
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, Keras, TensorFlow, NLTK  
- **Techniques:** Tokenization, sequence encoding, dropout regularization,  
  ensemble learning, stratified cross-validation  
- **Models:** Stacked LSTM networks, Random Forest classifiers, classical machine  
  learning algorithms  
- **Evaluation:** Accuracy scoring, confusion matrices, classification reports  
- **Visualization:** Data plotting, boxplots, image rendering  


## Capabilities Demonstrated

- Preprocessing and transforming textual and image data for machine learning workflows  
- Implementing deep recurrent neural networks for sequential data modeling  
- Applying ensemble tree-based classifiers for robust image recognition  
- Systematic benchmarking and performance evaluation of ML models  
- Clear communication of results through quantitative metrics and visualizations  
- Managing training checkpoints and reproducible experimentation  


## Use Cases

- Character-level text generation for language modeling tasks  
- Automated recognition of handwritten digits from pixel data  
- Benchmarking classical ML algorithms on tabular datasets  
- Foundations for building scalable NLP and computer vision applications  


## License

MIT License — Free to use, modify, and distribute.
