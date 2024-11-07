import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Stock_MARKET_PREDICTION/best_ensemble_model.h5")
    return model

# Load and preprocess the dataset
def load_data(file):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Remove dollar signs and convert to float
    data[' Close/Last'] = data[' Close/Last'].replace('[\$,]', '', regex=True).astype(float)
    data[' Open'] = data[' Open'].replace('[\$,]', '', regex=True).astype(float)
    data[' High'] = data[' High'].replace('[\$,]', '', regex=True).astype(float)
    data[' Low'] = data[' Low'].replace('[\$,]', '', regex=True).astype(float)

    # Feature engineering: add moving averages (SMA and EMA)
    data['SMA_50'] = data[' Close/Last'].rolling(window=50).mean()
    data['SMA_200'] = data[' Close/Last'].rolling(window=200).mean()
    data['EMA_50'] = data[' Close/Last'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data[' Close/Last'].ewm(span=200, adjust=False).mean()

    return data.dropna()

# Scale data and prepare sequences
def preprocess_data(data, features, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])
    
    sequences = []
    for i in range(sequence_length, len(scaled_data)):
        sequences.append(scaled_data[i-sequence_length:i])
    return np.array(sequences), scaler

# Predict future stock prices
def predict_stock(model, data, scaler):
    predictions = model.predict(data)
    
    # Create a placeholder for inverse transformation with the same shape as the original scaled data
    placeholder = np.zeros((predictions.shape[0], len(scaler.min_)))
    
    # Place predictions in the correct column (e.g., first column if predicting 'Close/Last')
    placeholder[:, 0] = predictions.flatten()  # Assume 'Close/Last' is the first feature in scaler
    
    # Inverse transform using the placeholder
    return scaler.inverse_transform(placeholder)[:, 0]  # Only take the 'Close/Last' column


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Dashboard", "About Us/Project Details", "Prediction"])

# Page 1: Dashboard
# Page 1: Dashboard
# Page 1: Dashboard
if page == "Dashboard":
    st.title("Stock Market Prediction")
     # Use raw string for the file path or ensure backslashes are doubled
    image_path = "Stock_MARKET_PREDICTION/Leonardo_Phoenix_A_sleek_and_modern_digital_illustration_of_a_2.jpg"
    
    # Display the image with a caption
    st.image(image_path, use_column_width=True)
   
    
    st.write("""
    ### Stock Price Prediction using LSTM and RNN Ensemble
    This project showcases the use of advanced deep learning architectures, specifically LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) layers, to forecast stock prices. While the model was initially trained on Apple stock data, you can upload your own dataset under the **Prediction** tab to generate personalized forecasts for any stock of your choice.

    Key Features:
    - **Data Preparation and Preprocessing**: The project utilizes a sliding window approach to create input sequences, effectively capturing temporal dependencies in stock prices.
    - **Flexible Model Architecture**: The `EnhancedModel` class allows for various RNN architectures, including LSTM, GRU, and Bidirectional LSTM layers, and customizable parameters.
    - **Hyperparameter Tuning**: KerasTuner’s RandomSearch automatically finds the best configuration to optimize model performance.
    - **Ensemble Approach**: Multiple models are trained and combined, leveraging the strengths of each model to improve accuracy and stability.
    - **Comprehensive Performance Evaluation**: The model’s performance is assessed using metrics such as RMSE, MAE, and R², ensuring a detailed accuracy analysis.
    - **Visualization**: Plots of actual vs. predicted prices, residual plots, and error distribution provide insights into model performance.

    ### Performance Metrics
    - **Ensemble Model RMSE**: 0.0123
    - **Ensemble Model MAE**: 0.0101
    - **Ensemble Model R²**: 0.972

    These metrics highlight the ensemble model’s effectiveness in achieving high prediction accuracy for stock price forecasting.
    """)
    st.write("###### Navigate to **About Us** to learn more about the project or **Prediction** to start forecasting with your own data.")



# Page 2: About Us/Project Details
elif page == "About Us/Project Details":
    st.title("About the Project")

    st.write("### Project Details")
    st.write("""
    This project utilizes advanced deep learning techniques, specifically Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, to predict stock prices based on historical trends. While initially trained on Apple stock data, this model supports predictions for other datasets as well.
    
    #### Dataset
    - The model is trained using a dataset of historical stock prices for Apple, `historicaldata.csv`, which includes key metrics such as open, close, high, and low prices over several years.
    - Using this dataset allows the model to learn from a range of market conditions, trends, and seasonal variations, enhancing its forecasting accuracy.

    #### Challenges
    Developing an accurate stock price prediction model presented several challenges:
    
    - **Volatility and Unpredictability**: Stock prices are inherently volatile and influenced by numerous external factors, making it difficult for models to achieve high accuracy consistently.
    - **Temporal Dependencies**: Capturing temporal dependencies in stock prices requires advanced feature engineering and sequence processing. A sliding window approach was used to create sequences for LSTM and GRU models, ensuring that the models learned patterns over different time periods.
    - **Hyperparameter Tuning**: Choosing the best model configuration (number of units, dropout rates, etc.) was crucial for optimizing performance. We used KerasTuner’s RandomSearch for automated hyperparameter tuning to find the most effective model setup.
    - **Overfitting**: Given the complexity of deep learning models, overfitting was a significant risk. Techniques like dropout layers, regularization, and ensemble modeling helped to mitigate this and improve the model's generalizability.
    - **Long Training Times**: The LSTM and GRU layers are computationally intensive, leading to longer training times. Using an ensemble of models further increased this training duration, but it ultimately improved the prediction accuracy and stability.

    This project demonstrates how combining advanced deep learning architectures and ensemble methods can address some of the common challenges in time series forecasting and improve prediction accuracy. The use of historical stock data as a training base enables users to make data-driven decisions and potentially gain valuable insights from market trends.
    """)

# Page 3: Prediction
elif page == "Prediction":
    st.title("Predict Future Stock Prices")

    # File uploader for user data input
    file = st.file_uploader("Upload your stock data CSV", type=["csv"])
    if file is not None:
        data = load_data(file)
        model = load_model()
        
        # Display recent data
        st.subheader("Recent Stock Data")
        st.write(data.tail(10))
        
        # Define features for prediction
        features = [' Close/Last', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200']
        
        # User-defined sequence length
        sequence_length = st.sidebar.slider("Sequence Length", min_value=30, max_value=120, value=60, step=10)
        sequences, scaler = preprocess_data(data, features, sequence_length=sequence_length)

        # Prediction and visualization
        predictions = predict_stock(model, sequences, scaler)
        
        st.subheader("Predicted vs. Actual Stock Prices")
        fig, ax = plt.subplots()
        ax.plot(data.index[-len(predictions):], data[' Close/Last'][-len(predictions):], label="Actual")
        ax.plot(data.index[-len(predictions):], predictions, label="Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.legend()
        st.pyplot(fig)

        # Future forecast option
        forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=5)
        last_sequence = sequences[-1].reshape(1, sequence_length, len(features))
        # Prepare future predictions array with correct shape for inverse transformation
        future_predictions_scaled = []

# Loop for forecasting future days
        for _ in range(forecast_days):
              prediction = model.predict(last_sequence)
              future_predictions_scaled.append(prediction[0, 0])

              # Placeholder for new prediction to be appended correctly with matching features
              new_prediction = np.zeros((1, 1, len(features)))
              new_prediction[0, 0, 0] = prediction  # Place prediction in the first column

    # Update last_sequence for the next prediction
              last_sequence = np.append(last_sequence[:, 1:, :], new_prediction, axis=1)

# Convert future_predictions to the original scale
        future_predictions_array = np.array(future_predictions_scaled).reshape(-1, 1)
        placeholder_array = np.zeros((future_predictions_array.shape[0], len(features)))
        placeholder_array[:, 0] = future_predictions_array.flatten()

# Use inverse_transform on the placeholder with correct dimensions
        future_predictions = scaler.inverse_transform(placeholder_array)[:, 0]

        st.subheader(f"Next {forecast_days} Days Forecast")
        # Set the Date column as the index if not already set
        data.set_index('Date', inplace=True)

# Calculate future dates based on the last date in the index
        last_date = data.index[-1] if isinstance(data.index[-1], pd.Timestamp) else pd.to_datetime(data.index[-1])
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

        forecast_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Forecast'])
        st.line_chart(forecast_df)

