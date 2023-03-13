import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title('Neural Network in Streamlit')

# Load the data
data = pd.read_csv('data.csv')

# Define the input and output columns
input_cols = ['feature1', 'feature2', 'feature3']
output_col = 'target'

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Separate the input and output columns
X_train = train_data[input_cols].values
y_train = train_data[output_col].values
X_test = test_data[input_cols].values
y_test = test_data[output_col].values

# Define the model architecture
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
st.write("Test Accuracy:", accuracy)
