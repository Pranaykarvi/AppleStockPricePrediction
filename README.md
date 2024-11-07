# Apple Stock Price Prediction  
## LSTM and RNN Ensemble for Time Series Forecasting

This project demonstrates the use of advanced deep learning architectures, specifically LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) layers, to forecast Apple stock prices. By implementing an ensemble of these models, the project achieves higher prediction accuracy. The notebook utilizes Keras and KerasTuner for hyperparameter tuning and model selection, making it flexible and adaptable for various time series forecasting tasks.

### Key Features
- **Data Preparation and Preprocessing**:  
  - The project starts by loading and scaling the data using `MinMaxScaler` for optimal model performance. 
  - A sliding window approach creates input sequences, where each sequence represents a specific time period of stock prices. This method effectively captures temporal dependencies.
  
- **EnhancedModel Class for Flexible Architectures**:  
  - The `EnhancedModel` class enables a variety of RNN architectures, including LSTM, GRU, and Bidirectional LSTM layers.  
  - It allows for customizable parameters such as the number of units, dropout rates, and layer types, making the model easily adaptable for other datasets or forecasting tasks.

- **Hyperparameter Tuning with KerasTuner**:  
  - KerasTuner’s `RandomSearch` automatically finds the best hyperparameter configuration based on validation loss, optimizing model performance.
  - Key tunable parameters include the number of LSTM/GRU units, dropout rates, learning rates, and the choice of RNN type.

- **Ensemble Approach**:  
  - An ensemble of multiple models is trained to improve predictive accuracy. Averaging the predictions of various architectures minimizes variance, leading to more stable and accurate forecasts.  
  - This approach outperforms individual models by leveraging the strengths of each model type.

- **Comprehensive Performance Evaluation**:  
  - The final model’s performance is evaluated with Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² metrics, which provide a detailed analysis of accuracy.
  - Comparison between individual and ensemble model results demonstrates the ensemble’s effectiveness.

- **Visualization**:  
  - Plots of actual vs. predicted stock prices are generated to visualize the model’s performance.
  - Additional visualizations, such as residual and error distribution plots, provide insights into prediction accuracy and potential areas for improvement.

### Performance Metrics
- **Ensemble Model RMSE**: 0.0123
- **Ensemble Model MAE**: 0.0101
- **Ensemble Model R²**: 0.972

These metrics underscore the ensemble’s effectiveness in achieving high prediction accuracy for stock price forecasting.

### Usage Instructions
1. **Data Preparation**: Ensure `X_train`, `X_val`, `y_train`, and `y_val` are defined with an appropriate time-series split.
2. **Model Configuration and Training**:  
   - Use the `EnhancedModel` class to experiment with different model architectures.
   - Run KerasTuner to identify the best hyperparameters.
3. **Evaluation and Visualization**:  
   - After training, evaluate the model using RMSE, MAE, and R² metrics.  
   - Plot actual vs. predicted values to validate forecasting accuracy.

### Dependencies
- Keras
- KerasTuner
- Scikit-Learn
- Matplotlib

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This project serves as a foundational template for time series forecasting using deep learning models. It is especially useful for those interested in experimenting with ensemble approaches in predictive modeling.

