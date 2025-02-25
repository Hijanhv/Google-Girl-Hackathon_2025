# RTL Combinational Depth Predictor

An AI-based tool to predict the combinational logic depth of signals in RTL designs without running full synthesis, helping to identify potential timing violations early in the design process.

## Problem Statement

Timing analysis is a crucial step in the design of any complex IP/SoC. However, timing analysis reports are generated after synthesis is complete, which is a very time-consuming process. This leads to overall delays in project execution time as timing violations can require architectural refactoring.

This tool uses machine learning to predict the combinational logic depth of signals in behavioral RTL, which can greatly speed up the timing analysis process.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) Icarus Verilog for RTL parsing

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/rtl-depth-predictor.git
cd rtl-depth-predictor
```

2. Create and activate the virtual environment:

```bash
# On Unix/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Create necessary directories:

```bash
mkdir -p models plots
```

## Usage

### Running the Complete Pipeline

The easiest way to run the pipeline is using the `run_pipeline.py` script:

```bash
# On Unix/macOS
# Train the model and compare different model types
python3 run_pipeline.py --train --compare_models

# On Windows
# Train the model and compare different model types
python run_pipeline.py --train --compare_models
```

For predicting depth or running tests:

```bash
# On Unix/macOS
# Predict depth for a signal in an RTL file
python3 run_pipeline.py --predict --rtl_file path/to/your/rtl_file.v --signal signal_name

# Run tests
python3 run_pipeline.py --test

# On Windows
# Predict depth for a signal in an RTL file
python run_pipeline.py --predict --rtl_file path/to/your/rtl_file.v --signal signal_name

# Run tests
python run_pipeline.py --test
```

### Training the Model

```bash
# On Unix/macOS
# Basic training
python3 src/train_model.py --data_path data/training_data.csv --test_data_path data/test_data.csv --model_output models/depth_predictor.joblib

# On Windows
# Basic training
python src/train_model.py --data_path data/training_data.csv --test_data_path data/test_data.csv --model_output models/depth_predictor.joblib
```

For comparing different model types:

```bash
# On Unix/macOS
python3 src/train_model.py --data_path data/training_data.csv --test_data_path data/test_data.csv --model_output models/depth_predictor.joblib --compare_models --plot_results

# On Windows
python src/train_model.py --data_path data/training_data.csv --test_data_path data/test_data.csv --model_output models/depth_predictor.joblib --compare_models --plot_results
```

### Predicting Combinational Depth

```bash
# On Unix/macOS
python3 src/predict_depth.py --rtl_file path/to/your/rtl_file.v --signal signal_name --model_path models/depth_predictor.joblib

# On Windows
python src/predict_depth.py --rtl_file path/to/your/rtl_file.v --signal signal_name --model_path models/depth_predictor.joblib
```

### Evaluating the Model

```bash
# On Unix/macOS
# Evaluate on test data
python3 evaluate_model.py

# Generate detailed visualizations
python3 visualize_results.py

# On Windows
# Evaluate on test data
python evaluate_model.py

# Generate detailed visualizations
python visualize_results.py
```

### Running Tests

```bash
# On Unix/macOS
python3 -m unittest discover tests

# On Windows
python -m unittest discover tests
```

### Running the Complete Pipeline with Shell Script

On Unix/macOS:

```bash
bash run_all.sh
```

On Windows (using Git Bash or WSL):

```bash
bash run_all.sh
```

## Project Structure

```
rtl_depth_predictor/
├── data/                  # Training and test datasets
│   ├── training_data.csv  # Training dataset
│   ├── test_data.csv      # Test dataset
│   ├── sample_rtl_1.v     # Sample RTL file (counter)
│   ├── sample_rtl_2.v     # Sample RTL file (alu)
│   └── sample_rtl_3.v     # Sample RTL file (fifo_controller)
├── models/                # Trained model files
│   └── depth_predictor.joblib  # Saved model
├── notebooks/             # Jupyter notebooks for exploration
│   └── model_exploration.ipynb
├── src/                   # Source code
│   ├── feature_extraction.py  # RTL feature extraction
│   ├── model.py           # ML model definition
│   ├── train_model.py     # Model training script
│   └── predict_depth.py   # Prediction script
├── tests/                 # Test files
│   └── test_model.py      # Model tests
├── evaluate_model.py      # Model evaluation script
├── visualize_results.py   # Visualization script
├── run_pipeline.py        # Pipeline execution script
├── run_all.sh             # Shell script to run complete pipeline
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Model Performance

The Ridge Regression model performs best on our dataset with the following metrics:

- MSE: 0.8135
- MAE: 0.6238
- R²: 0.5119
- Within ±1 depth level: 77.78%

## Approach

1. **Data Collection**: Generate synthetic RTL designs and extract their combinational depth using synthesis tools.
2. **Feature Engineering**: Extract relevant features from RTL code that influence combinational depth.
3. **Model Selection**: Compare different ML algorithms to find the best predictor.
4. **Training**: Train the selected model on the dataset.
5. **Evaluation**: Evaluate the model's accuracy and performance.

## License

MIT
