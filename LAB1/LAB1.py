import numpy as np
import matplotlib.pyplot as plt

def show_result(x, y, pred_y, losses) :
    # ground truth
    plt.subplot(1, 3, 1)
    plt.title('Ground truth', fontsize = 18)
    plt.xlabel("x", fontsize = 12)
    plt.ylabel("y", fontsize = 12)
    for i in range(x.shape[0]) :
        if y[i] == 0 :
            plt.plot(x[i][0], x[i][1], 'ro')
        else :
            plt.plot(x[i][0], x[i][1], 'bo')
    # predict result
    plt.subplot(1, 3, 2)
    plt.title('Predict result', fontsize = 18)
    plt.xlabel("x", fontsize = 12)
    plt.ylabel("y", fontsize = 12)
    for i in range(x.shape[0]) : 
        if pred_y[i] == 0 :
            plt.plot(x[i][0], x[i][1], 'ro')
        else :
            plt.plot(x[i][0], x[i][1], 'bo')
    # learning curve
    plt.subplot(1, 3, 3)
    plt.title('Learning curve', fontsize = 18)
    plt.xlabel("Epoch", fontsize = 12)
    plt.ylabel("Loss", fontsize = 12)
    plt.plot(losses)
    plt.tight_layout()
    plt.show()

def sigmoid(x) :
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x) :
    return x * (1 - x)

def generate_linear(n=100) :
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts :
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1] :
            labels.append(0)
        else :
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy() :
    inputs = []
    labels = []
    for i in range(11) :
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5 :
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def MSE(error) :
    loss = 0
    error = error.flatten()
    for j in range(error.size) :
        loss += error[j] ** 2
    return loss / error.size

def derivative_MSE(error) :
    return 2 * error

def train(learning_rate, neurons1, neurons2, mode, iteration, epoch) :
    # weights are small random values
    losses = []
    weight1 = np.random.random((2, neurons1))
    weight2 = np.random.random((neurons1, neurons2))
    weight3 = np.random.random((neurons2, 1))

    for i in range(epoch) :
        # choose input
        if mode == 0 :
            train_input, train_output = generate_XOR_easy()
        else :
            train_input, train_output = generate_linear(n=100)

        for j in range(iteration) :
            # forward pass
            layer1_output = sigmoid(np.dot(train_input, weight1))
            layer2_output = sigmoid(np.dot(layer1_output, weight2))
            network_output = sigmoid(np.dot(layer2_output, weight3))

            # back propagation
            # compute gradient of output & weight3
            error = network_output - train_output
            network_output_grad = derivative_MSE(error) * derivative_sigmoid(network_output)
            weight3_grad = np.dot(layer2_output.T, network_output_grad)

            # compute input back to layer2 & output of layer2
            layer2_back_input = np.dot(network_output_grad, weight3.T)
            layer2_back_output = layer2_back_input * derivative_sigmoid(layer2_output)
            # compute gradient of weight2
            weight2_grad = np.dot(layer1_output.T, layer2_back_output)

            # compute input back to layer1 & output of layer1
            layer1_back_input = np.dot(layer2_back_output, weight2.T)
            layer1_back_output = layer1_back_input * derivative_sigmoid(layer1_output)
            # compute gradient of weight1
            weight1_grad = np.dot(train_input.T, layer1_back_output)

            # weight update
            weight1 -= learning_rate * weight1_grad
            weight2 -= learning_rate * weight2_grad
            weight3 -= learning_rate * weight3_grad

            # compute loss
            loss = MSE(error)
        losses.append(loss)
        print("epoch " + str(i) + " loss : " + '%.6f' % loss)

    return weight1, weight2, weight3, np.array(losses)

if __name__ == "__main__" :
    # initialize
    learning_rate = 0.025
    neurons_layer1 = 5
    neurons_layer2 = 3
    epochs = 400
    iterations = 80
    np.random.seed(1)


    # XOR
    # training
    weight1, weight2, weight3, losses = train(learning_rate, neurons_layer1, neurons_layer2, 0, iterations, epochs)
    
    # testing
    ground_truth, pred_output = generate_XOR_easy()
    layer1_output_XOR = sigmoid(np.dot(ground_truth, weight1))
    layer2_output_XOR = sigmoid(np.dot(layer1_output_XOR, weight2))
    network_output_XOR = sigmoid(np.dot(layer2_output_XOR, weight3))

    # output
    correct = 0
    print(network_output_XOR.round(3))
    for i in range(pred_output.size) :
        if network_output_XOR[i].round(0) == pred_output[i] :
            correct += 1
    accuracy = (correct / pred_output.size) * 100
    print("accuracy is " + str(accuracy) + "%")
    show_result(ground_truth, pred_output, np.around(network_output_XOR), losses)
    

    # linear
    # training
    weight1, weight2, weight3, losses = train(learning_rate, neurons_layer1, neurons_layer2, 1, iterations, epochs)
    
    # testing
    ground_truth, pred_output = generate_linear(n=100)
    layer1_output_linear = sigmoid(np.dot(ground_truth, weight1))
    layer2_output_linear = sigmoid(np.dot(layer1_output_linear, weight2))
    network_output_linear = sigmoid(np.dot(layer2_output_linear, weight3))

    # output
    correct = 0
    print(network_output_linear.round(3))
    for i in range(pred_output.size) :
        if network_output_linear[i].round(0) == pred_output[i] :
            correct += 1
    accuracy = (correct / pred_output.size) * 100
    print("accuracy is " + str(accuracy) + "%")
    show_result(ground_truth, pred_output, np.around(network_output_linear), losses)