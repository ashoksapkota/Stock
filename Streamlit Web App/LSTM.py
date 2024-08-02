from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout # type: ignore 
import time
import tensorflow as tf

import plotly.graph_objects as go


from typing import TypeVar
T = TypeVar('T')


import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os

from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Bidirectional, LayerNormalization # type: ignore
from tensorflow.keras.optimizers import RMSprop # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




class LSTM_Model:
    def __init__(self, file_path, lag_stock=30):
        self.file_path = file_path
        self.lag_stock = lag_stock
        self.df_stock = None
        self.scaler = StandardScaler()
        self.model = None

    def load_data(self):
        self.df_stock = pd.read_csv(self.file_path, parse_dates=['Date'], index_col='Date')
        return self.df_stock

    def prepare_lagged_features(self):
        # Adding technical indicators here can improve accuracy
        for column in self.df_stock.select_dtypes(include=[np.number]).columns:
            for i in range(1, self.lag_stock + 1):
                self.df_stock[f"{column}(t-{i})"] = self.df_stock[column].shift(i)
        self.df_stock.dropna(inplace=True)
        return self.df_stock

    def scale_data(self):
        numeric_df_stock = self.df_stock.select_dtypes(include=[np.number])
        scaled_data = self.scaler.fit_transform(numeric_df_stock)
        return scaled_data

    def split_data(self, scaled_data):
        X, y = [], []
        for i in range(self.lag_stock, len(scaled_data)):
            X.append(scaled_data[i - self.lag_stock:i])
            y.append(scaled_data[i, 0])  # Assuming the target variable is the first column
        X, y = np.array(X), np.array(y)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, input_shape):
        model = Sequential()
        
        model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(input_shape[0], input_shape[1])))
        model.add(LayerNormalization())
        model.add(Dropout(0.3))
        
        model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
        model.add(LayerNormalization())
        model.add(Dropout(0.3))
        
        model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
        model.add(LayerNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1))
        
        optimizer = RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_squared_error'])
        
        self.model = model
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 20))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_scheduler]
        )
        return history

    def predict_next_30_days(self, last_sequence):
        predictions = []
        current_sequence = last_sequence

        for _ in range(30):
            prediction = self.model.predict(current_sequence[np.newaxis, :, :])
            predictions.append(prediction[0, 0])

            new_value = np.array([[prediction[0, 0]] * current_sequence.shape[1]])
            current_sequence = np.append(current_sequence[1:], new_value, axis=0)
        
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_full_data = np.zeros((predictions.shape[0], self.scaler.mean_.shape[0]))
        dummy_full_data[:, 0] = predictions[:, 0]
        predictions = self.scaler.inverse_transform(dummy_full_data)[:, 0]

        return predictions

    def plot_predictions(self, predictions):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predictions'))
        fig.update_layout(title='30-Day Stock Price Predictions', xaxis_title='Day', yaxis_title='Price')
        st.plotly_chart(fig)

    def show_predictions_table(self, predictions):
        dates = pd.date_range(start=pd.Timestamp.today(), periods=len(predictions), freq='D')
        predictions_df = pd.DataFrame({'Date': dates, 'Predicted Price': predictions})
        return predictions_df

    def get_stock_data(self, ticker):
        file = f'{ticker}.csv'
        self.df_stock = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
        self.df_stock.index = self.df_stock.index.date
        st.write(f'Training Selected Machine Learning models for {ticker}')
        st.markdown('Your **_final_ _dataframe_ _for_ Training** ')
        st.write(self.df_stock)
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        st.success('Training Completed!')
        return self.df_stock
