import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

####################
### Set the hyperparameters in you Network.py file ###
####################

from Network import iterations, learning_rate, hidden_nodes, output_nodes, Network
from prepareData import train_features, train_targets, val_targets, val_features, scaled_features, test_features, test_targets, rides, test_data


def MSE(y, Y):
    return np.mean((y-Y)**2)

N_i = train_features.shape[1]
network = Network(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
plot = 1

# TRAINING NN
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['cnt']

    network.train(X, y)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii / float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


# VISUALIZE TRAINING INFO
if (plot):
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    _ = plt.ylim()

    fig, ax = plt.subplots(figsize=(8,4))

    mean, std = scaled_features['cnt']
    predictions = network.run(test_features).T*std + mean
    ax.plot(predictions[0], label='Prediction')
    ax.plot((test_targets['cnt']*std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.iloc[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)

    plt.show()