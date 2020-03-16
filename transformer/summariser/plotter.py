import plotly.express as px
import pickle
import pandas as pd

training_loss = []
training_accuracy = []

with open('/home/joe/GitDrive/AI-Portfolio/transformer/summariser/training_results.txt', 'rb') as f:
    training_loss = pickle.load(f)
    training_accuracy = pickle.load(f)

loss_df = pd.DataFrame(training_loss, columns=['epoch', 'batch', 'loss'])
pd.to_numeric(loss_df['loss'])
accuracy_df = pd.DataFrame(training_accuracy, columns=['epoch', 'batch', 'accuracy'])
pd.to_numeric(accuracy_df['accuracy'])

fig1 = px.line(loss_df, x='batch', y='loss', color='epoch')
fig2 = px.line(accuracy_df, x='batch', y='accuracy', color='epoch')

fig1.show()
fig2.show()