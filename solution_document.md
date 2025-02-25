# RTL Combinational Depth Predictor - Solution Document

## Project Information

- **Name**: Janhavi
- **Email**: janhavi@example.com
- **GitHub**: janhavi-gh

## Problem Statement

Timing analysis is a crucial step in the design of any complex IP/SoC. However, timing analysis reports are generated after synthesis is complete, which is a very time-consuming process. This leads to overall delays in project execution time as timing violations can require architectural refactoring.

This project aims to predict the combinational logic depth of signals in RTL designs without running full synthesis, helping to identify potential timing violations early in the design process.

## Approach

1. **Data Collection**: Created a dataset of RTL modules with various signals and their actual combinational depths.
2. **Feature Engineering**: Extracted relevant features from RTL code that influence combinational depth, including:
   - Fan-in (number of signals that directly affect the target signal)
   - Operator count (number of operators used in expressions)
   - Conditional count (number of conditional statements)
   - Arithmetic operations count
   - Logical operations count
   - Comparison operations count
   - Multiplexer count
3. **Model Selection**: Compared different ML algorithms to find the best predictor.
4. **Training**: Trained the selected model on the dataset.
5. **Evaluation**: Evaluated the model's accuracy and performance.

## Implementation

### Project Structure

```
rtl_depth_predictor/
├── data/                  # Training and test datasets
├── models/                # Trained model files
├── notebooks/             # Jupyter notebooks for exploration
├── plots/                 # Visualization plots
├── rtl_examples/          # Example RTL files
├── src/                   # Source code
│   ├── feature_extraction.py  # RTL feature extraction
│   ├── model.py           # ML model definition
│   ├── train_model.py     # Model training script
│   └── predict_depth.py   # Prediction script
├── tests/                 # Test files
├── evaluate_model.py      # Model evaluation script
├── run_pipeline.py        # Pipeline execution script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

### Feature Extraction

The feature extraction module (`src/feature_extraction.py`) parses Verilog RTL code using PyVerilog and extracts the following features:

1. **Fan-in**: Number of signals that directly affect the target signal.
2. **Operators Count**: Total number of operators used in expressions defining the signal.
3. **Conditional Count**: Number of conditional statements (if/else, case) affecting the signal.
4. **Arithmetic Operations**: Count of arithmetic operators (+, -, \*, /, etc.).
5. **Logical Operations**: Count of logical operators (&, |, ^, ~, etc.).
6. **Comparison Operations**: Count of comparison operators (==, !=, <, >, etc.).
7. **Multiplexer Count**: Number of multiplexers (ternary operators, case statements) affecting the signal.

### Model Selection

We compared several machine learning models:

- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (SVR)
- Multi-layer Perceptron (MLP)

After evaluation, the **Ridge Regression** model performed best on our dataset with the following metrics:

- MSE: 0.8135
- MAE: 0.6238
- R²: 0.5119
- Within ±1 depth level: 77.78%

## Results

### Model Performance

The model was evaluated on a test dataset containing 18 different signals from various RTL modules. The results show that:

1. The model achieves an R² score of 0.5119, indicating it explains about 51% of the variance in combinational depth.
2. The mean absolute error (MAE) is 0.6238, meaning predictions are off by less than 1 depth level on average.
3. 77.78% of predictions are within ±1 of the actual depth, which is sufficient for early timing violation detection.

### Prediction Examples

Here are some examples of the model's predictions:

| Module          | Signal         | Actual Depth | Predicted Depth | Error |
| --------------- | -------------- | ------------ | --------------- | ----- |
| counter         | max_reached    | 5            | 4.97            | 0.03  |
| alu             | overflow_flag  | 7            | 5.54            | 1.46  |
| fifo_controller | debug_trigger  | 7            | 4.79            | 2.21  |
| ethernet        | packet_valid   | 5            | 4.99            | 0.01  |
| processor       | pipeline_stall | 5            | 4.73            | 0.27  |

### Challenges and Limitations

1. **Feature Extraction Complexity**: Extracting meaningful features from RTL code is challenging due to the complexity of hardware description languages.
2. **Limited Dataset**: The model's performance could be improved with a larger and more diverse dataset.
3. **Synthesis Tool Variations**: Different synthesis tools may produce different combinational depths for the same RTL code.
4. **Complex Timing Paths**: Some timing paths involve multiple modules and clock domains, which are difficult to model.

## Future Improvements

1. **Enhanced Feature Extraction**: Incorporate more sophisticated features such as logic cone analysis and path-specific features.
2. **Larger Dataset**: Expand the dataset with more diverse RTL designs and synthesis results.
3. **Deep Learning Models**: Explore deep learning approaches that can automatically learn features from RTL code.
4. **Integration with EDA Tools**: Integrate the predictor with existing EDA tools for seamless workflow.
5. **Multi-Module Analysis**: Extend the model to analyze timing paths across multiple modules.

## Conclusion

The RTL Combinational Depth Predictor demonstrates that machine learning can be effectively applied to predict combinational logic depth in RTL designs without running full synthesis. With an accuracy of 77.78% (predictions within ±1 of actual depth), the tool can help identify potential timing violations early in the design process, reducing the overall project execution time.

The Ridge Regression model provides a good balance between accuracy and interpretability, making it suitable for this application. The feature importance analysis shows that fan-in, operator count, and logical operations are the most significant predictors of combinational depth.

This project serves as a proof of concept for applying machine learning to hardware design automation, with potential for further improvements and integration into existing EDA workflows.
