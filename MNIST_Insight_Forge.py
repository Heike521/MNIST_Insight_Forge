# Import required modules
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw, ImageFont
# Set constants
# Number of passes
EPOCHS = 5
# Learning rate
LEARNING_RATE = 0.5
# Definition of individual nodes
# Input nodes
INPUT_NODES = 784
# Hidden nodes
HIDDEN_NODES = 300
# Output nodes
OUTPUT_NODES = 10
# Compute initial weights
WEIGHT_INPUT_HIDDEN = np.random.uniform(-0.5, 0.5, (INPUT_NODES, HIDDEN_NODES))
WEIGHT_HIDDEN_OUTPUT = np.random.uniform(-0.5, 0.5, (HIDDEN_NODES, OUTPUT_NODES))
# Bias vectors for input and output layers
BIAS_HIDDEN = np.zeros((HIDDEN_NODES,))
BIAS_OUTPUT = np.zeros((OUTPUT_NODES,))
# Constants for visualization
NUMBER = 10
# Batch size for training
BATCH_SIZE = 32
# Set random seed for reproducibility
np.random.seed(42)
# Create output folder for all results
OUTPUT_DIR = "Projektarbeit_Output_Bilder"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Definition of individual classes
class MNISTDownLoad:
    ''' Class for downloading the complete MNIST dataset via torchvision.

        It combines training and test data, allows a flexible split into
        training and test sets, and stores the data locally in .npz format
        for offline use.
    '''

    def __init__(self, test_size: float = 0.2, random_state: int = 42,
                 data_way: str = "mnist_data"):
        '''
        Initialize the class for loading and splitting the MNIST dataset.

        Parameters
        ----------
        test_size : float
            Share of test data (between 0.0 and 1.0).
        random_state : int
            Seed for the random number generator (reproducibility).
        data_way : str
            Directory where the data will be saved.
        '''
        self.test_size = test_size
        self.random_state = random_state
        self.data_way = data_way
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_and_split_mnist_data(self):
        '''
        Load the MNIST dataset, combine training and test data, and split
        them into training and test sets according to the given test size.
        '''
        print("Loading the full MNIST dataset ...... ")
        train_set = datasets.MNIST(root=self.data_way, train=True,
                                   download=True, transform=ToTensor())
        test_set = datasets.MNIST(root=self.data_way, train=False,
                                  download=True, transform=ToTensor())
        # Convert tensor data to NumPy arrays
        X_train_raw = train_set.data.numpy()
        y_train_raw = train_set.targets.numpy()
        X_test_raw = test_set.data.numpy()
        y_test_raw = test_set.targets.numpy()
        # Merge both datasets
        X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
        # Split into training and test data per test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_all, y_all, test_size=self.test_size,
            random_state=self.random_state, stratify=y_all,)
        print("Split into training and test sets completed")

    def save_mnist_data_new(self, way: str = "mnist_daten_split.npz"):
        '''
        Save the split training and test data as a compressed .npz file.

        Parameters
        ----------
        way : str
            Path to the output file (.npz).
        '''
        try:
            if self.X_train is None or self.X_test is None:
                raise ValueError('''Data not loaded. Please run function:
                            load_and_split_mnist_data again''')
            way = os.path.abspath(way)
            np.savez_compressed(way, X_train=self.X_train,
                                y_train=self.y_train, X_test=self.X_test,
                                y_test=self.y_test)
            print(f"Data saved to {way}")
        except Exception as e:
            print(f"Error while saving data: {e}")

    def load_mnist_data_npz(self, way: str):
        '''
        Load training and test data from an existing .npz file.

        Parameters
        ----------
        way : str
            Path to the saved .npz file.
        '''
        try:
            way = os.path.abspath(way)
            data = np.load(way)
            self.X_train = data["X_train"]
            self.y_train = data["y_train"]
            self.X_test = data["X_test"]
            self.y_test = data["y_test"]
            print(f"Data successfully loaded from: {way}")
        except Exception as e:
            print(f"File not found: {e}")


class NeuralNetTraining:
    '''
    Define and train a simple artificial neural network with a single hidden
    layer. Provide methods for forward and backpropagation, error calculation,
    and training evaluation.
    '''
    def __init__(self):
        '''
        Construct the class and initialize a neural network. All parameters
        come from the globally defined constants above.
        '''
        self.epochs = EPOCHS
        self.learning_rate = LEARNING_RATE
        self.input_nodes = INPUT_NODES
        self.hidden_nodes = HIDDEN_NODES
        self.output_nodes = OUTPUT_NODES
        # Copy the global constants for modification and use
        self.weight_input_hidden = WEIGHT_INPUT_HIDDEN.copy()
        self.weight_hidden_output = WEIGHT_HIDDEN_OUTPUT.copy()
        # Bias vectors for input and output layers
        # Bias for the hidden layer, as a copy of the global constant
        self.bias_hidden = BIAS_HIDDEN.copy()
        # Bias for the output layer, as a copy of the global constant
        self.bias_output = BIAS_OUTPUT.copy()

    def sig(self, x: float | np.ndarray) -> float | np.ndarray:
        '''
        Compute the sigmoid activation function.

        Parameters
        ----------
        x : float or np.ndarray
            Input value(s)

        Returns
        -------
        float or np.ndarray
            Activation value(s)
        '''
        return 1 / (1 + np.exp(-x))

    def sig_der(self, y) -> float | np.ndarray:
        '''
        Compute the derivative of the sigmoid function for backpropagation.

        Parameters
        ----------
        y : float or np.ndarray
            Output value(s) of the sigmoid function

        Returns
        -------
        float or np.ndarray
            Derivative value(s)
        '''
        return y * (1 - y)

    def training_batch(self, X_batch, y_batch):
        '''
        Perform a single training pass (forward and backpropagation) for
        one batch.

        Parameters
        ----------
        X_batch : np.ndarray
            Input data (batch)
        y_batch : np.ndarray
            Target outputs (one-hot encoded)

        Returns
        -------
        float
            Mean squared error of the batch
        '''
        # Forward propagation
        hidden_input = np.dot(X_batch, self.weight_input_hidden) + self.bias_hidden
        hidden_output = self.sig(hidden_input)
        output_input = np.dot(hidden_output, self.weight_hidden_output) + self.bias_output
        output_data = self.sig(output_input)
        # Error calculation
        error_output = y_batch - output_data
        error_hidden = np.dot(error_output * self.sig_der(output_data),
                              self.weight_hidden_output.T) * self.sig_der(hidden_output)
        # Update weights (backpropagation)
        self.weight_hidden_output += self.learning_rate * np.dot(hidden_output.T,
                                                                 error_output * self.sig_der(output_data)) / X_batch.shape[0]
        self.weight_input_hidden += self.learning_rate * np.dot(X_batch.T,
                                                                error_hidden) / X_batch.shape[0]
        # Update bias vectors
        self.bias_output += self.learning_rate * np.mean(error_output * self.sig_der(output_data), axis=0)
        self.bias_hidden += self.learning_rate * np.mean(error_hidden, axis=0)
        return np.mean(np.square(error_output))

    def predict(self, X,):
        '''
        Run a forward pass for arbitrary input.

        Parameters
        ----------
        X : np.ndarray
            Input data (single image or array of images)

        Returns
        -------
        np.ndarray
            Network outputs for each input
        '''
        # If a single image comes as a 1D vector, reshape to (1, 784)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Add bias in the forward pass
        H_predict = self.sig(np.dot(X, self.weight_input_hidden) + self.bias_hidden)
        O_predict = self.sig(np.dot(H_predict, self.weight_hidden_output) + self.bias_output)
        return O_predict

    def training_epochs(self, X_train, y_train, X_val=None, y_val=None):
        '''
        Train the neural network over multiple epochs and return loss and
        accuracy history.

        Parameters
        ----------
        X_train : np.ndarray
            Training images
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation images
        y_val : np.ndarray, optional
            Validation targets

        Returns
        -------
        Tuple[List[float], List[float or None]]
            Loss per epoch, accuracy per epoch (if val data provided)
        '''
        n_samples = X_train.shape[0]
        y_train_oh = np.eye(self.output_nodes)[y_train]
        error_list = []
        acc_list = []
        for ep in range(self.epochs):
            idx = np.random.permutation(n_samples)
            X_shuf = X_train[idx]
            y_shuf = y_train_oh[idx]
            batch_losses = []
            for start in range(0, n_samples, BATCH_SIZE):
                end = start + BATCH_SIZE
                X_batch = X_shuf[start:end].reshape(-1, self.input_nodes) / 255.0
                Y_batch = y_shuf[start:end].reshape(-1, self.output_nodes)
                batch_loss = self.training_batch(X_batch, Y_batch)
                batch_losses.append(batch_loss)
            ep_loss = np.mean(batch_losses)
            error_list.append(ep_loss)
            # Optional: accuracy on val set per epoch
            if X_val is not None and y_val is not None:
                y_pred = np.argmax(self.predict(X_val.reshape(-1,
                                       self.input_nodes) / 255.0), axis=1)
                acc = np.mean(y_pred == y_val)
                acc_list.append(acc)
            else:
                acc_list.append(None)
            if (ep + 1) % 10 == 0 or ep == self.epochs - 1:
                print(f"Epoch {ep+1}/{self.epochs} completed, "
                      f"loss: {ep_loss:.4f}")
        return error_list, acc_list

    def accuracy(self, X, y):
        '''
        Compute classification accuracy for given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray
            True targets

        Returns
        -------
        float
            Accuracy (share of correctly classified examples)
        '''
        preds = self.predict(X.reshape(-1, self.input_nodes) / 255.0)
        y_pred = np.argmax(preds, axis=1)
        return np.mean(y_pred == y)


class Visualizer:
    '''
    Visualize the training and test results of a neural network. Provide methods
    to save sample data, loss curves, accuracy curves, weight matrices, and
    misclassified images.
    '''

    def __init__(self, X_train, y_train, X_test, y_test):
        '''
        Initialize the visualization instance with training and test data.

        Parameters
        ----------
        X_train, y_train, X_test, y_test : np.ndarray
            Datasets to be used for visualizations.
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_sample_picture_train(self, data_way="mnist_bsp_train"):
        '''
        Save example images with labels from the training/test set as PNG files.

        Parameters
        ----------
        data_way : str
            Directory prefix for the output files.
        '''
        # Ensure folder exists
        os.makedirs(data_way, exist_ok=True)

        for i in range(NUMBER):
            # Create PIL image from training data
            picture = Image.fromarray(self.X_train[i].astype(np.uint8))
            # Enlarge image
            picture_train_bigger = picture.resize((115, 115),
                                                  Image.Resampling.NEAREST)
            # Save image
            filename = f"trainbild_{i}_label_{self.y_train[i]}.png"
            way = os.path.join(data_way, filename)
            picture_train_bigger.save(way)
        print(f"Saved {NUMBER} training sample images")

    def save_sample_picture_test(self, data_way="mnist_bsp_test"):
        '''
        Save example images with labels from the test set as PNG files.

        Parameters
        ----------
        data_way : str
            Directory prefix for the output files.
        '''
        # Ensure folder exists
        os.makedirs(data_way, exist_ok=True)
        for i in range(NUMBER):
            picture = Image.fromarray(self.X_test[i].astype(np.uint8))
            picture_test_bigger = picture.resize((115, 115),
                                                 Image.Resampling.NEAREST)
            # Build path
            filename = f"testbild_{i}_label_{self.y_test[i]}.png"
            way = os.path.join(data_way, filename)

            picture_test_bigger.save(way)
        print(f"Saved {NUMBER} test sample images")

    def save_error_curve(self, error_list, data_way="fehlerkurve.png"):
        '''
        Draw the loss curve over the training epochs and save it as an image.

        Parameters
        ----------
        error_list : List[float]
            Loss values per epoch.
        data_way : str
            Output path for the graphic.
        '''
        wide, hight, = 480, 320
        picture_error_curve = Image.new("RGB", (wide, hight), "white")
        draw = ImageDraw.Draw(picture_error_curve)
        # Axes x, y
        draw.line((40, hight-40, wide-20, hight-40), fill="black")
        draw.line((40, hight-40, 40, 20), fill="black")
        # Plot values
        if len(error_list) > 1:
            max_error = max(error_list)
            min_error = min(error_list)
            n = len(error_list)
            scale_x = (wide-60) / (n-1)
            scale_y = (hight-60) / (max_error - min_error + 1e-8)
            # Draw loss curve
            for i in range(n-1):
                x1 = 40 + i * scale_x
                y1 = hight - 40 - (error_list[i] - min_error) * scale_y
                x2 = 40 + (i + 1) * scale_x
                y2 = hight - 40 - (error_list[i + 1] - min_error) * scale_y
                draw.line((x1, y1, x2, y2), fill="red", width=2)
        picture_error_curve.save(data_way)
        print(f"Saved loss curve as {data_way}")

    def save_accuracy(self, acc_list, data_way="genauigkeitskurve.png"):
        '''
        Draw the accuracy curve over the training/validation epochs.

        Parameters
        ----------
        acc_list : List[float or None]
            Accuracy values per epoch.
        data_way : str
            Output path for the graphic.
        '''
        wide, hight = 480, 320
        picture_accuracy = Image.new("RGB", (wide, hight), "white")
        draw = ImageDraw.Draw(picture_accuracy)
        draw.line((40, hight-40, wide-20, hight-40), fill="black")
        draw.line((40, hight-40, 40, 20), fill="black")
        if len(acc_list) > 1:
            valid = [v for v in acc_list if v is not None]
            if valid:
                max_acc = max(acc_list)
                min_acc = min(acc_list)
            else:
                print("No valid accuracy values available.")
                return
            n = len(acc_list)
            scale_x = (wide-60) / (n-1)
            scale_y = (hight-60) / (max_acc - min_acc + 1e-8)
            for i in range(n-1):
                x1 = 40 + i * scale_x
                y1 = hight - 40 - (acc_list[i] - min_acc) * scale_y
                x2 = 40 + (i + 1) * scale_x
                y2 = hight - 40 - (acc_list[i + 1] - min_acc) * scale_y
                draw.line((x1, y1, x2, y2), fill="red", width=2)
        picture_accuracy.save(data_way)
        print(f"Saved accuracy curve as {data_way}")

    def save_train_vs_test_curve(self, error_list, acc_list,
                                 data_way="train_vs_test.png"):
        '''
        Draw the loss and accuracy curves for training and test in a single
        image. Overplot the curves (left y-axis = loss, right y-axis = accuracy).

        Parameters
        ----------
        error_list : List[float]
            Training loss values per epoch.
        acc_list : List[float or None]
            Accuracy values per epoch (training vs. test).
        data_way : str
            Output path for the graphic.
        '''
        wide, hight = 640, 400
        picture = Image.new("RGB", (wide, hight), "white")
        draw = ImageDraw.Draw(picture)
        # Axes
        draw.line((60, hight-60, wide-30, hight-60), fill="black")
        draw.line((60, hight-60, 60, 30), fill="black")
        # Loss curve left (red), accuracy curve right (blue)
        if len(error_list) > 1:
            n = len(error_list)
            max_error = max(error_list)
            min_error = min(error_list)
            max_acc = max([v for v in acc_list if v is not None], default=1)
            min_acc = min([v for v in acc_list if v is not None], default=0)
            scale_x = (wide-90) / (n-1)
            scale_y_err = (hight-90) / (max_error - min_error + 1e-8)
            scale_y_acc = (hight-90) / (max_acc - min_acc + 1e-8)
            # Loss curve (red, left)
            for i in range(n-1):
                x1 = 60 + i * scale_x
                y1 = hight-60 - (error_list[i] - min_error) * scale_y_err
                x2 = 60 + (i+1) * scale_x
                y2 = hight-60 - (error_list[i+1] - min_error) * scale_y_err
                draw.line((x1, y1, x2, y2), fill="red", width=2)
            # Accuracy curve (blue, right)
            for i in range(n-1):
                if acc_list[i] is not None and acc_list[i+1] is not None:
                    x1 = 60 + i * scale_x
                    y1 = hight-60 - (acc_list[i] - min_acc) * scale_y_acc
                    x2 = 60 + (i+1) * scale_x
                    y2 = hight-60 - (acc_list[i+1] - min_acc) * scale_y_acc
                    draw.line((x1, y1, x2, y2), fill="blue", width=2)
            # Legend
            draw.text((wide-200, 40), "Loss (red, left)", fill="red")
            draw.text((wide-200, 60), "Accuracy (blue, right)", fill="blue")
        picture.save(data_way)
        print(f"Saved training/test curve as {data_way}")

    def save_weights(self, weight_input_hidden, NUMBER,
                     data_way="gewichte"):
        '''
        Visualize the first NUMBER weights of the input layer as images.

        Parameters
        ----------
        weight_input_hidden : np.ndarray
            Weight matrix (784 x HIDDEN_NODES).
        NUMBER : int
            Number of weight images to save.
        data_way : str
            Target path for the resulting image.
        '''
        # Ensure folder exists
        os.makedirs(data_way, exist_ok=True)
        w_picture = []
        for i in range(NUMBER):
            picture = weight_input_hidden[:, i].reshape(28, 28)
            # Normalize image
            picture = 255 * (picture - picture.min()) / (np.ptp(picture) + 1e-8)
            picture_unit = Image.fromarray(picture.astype(np.uint8)).resize((56, 56),
                                                                            Image.Resampling.NEAREST)
            w_picture.append(picture_unit)
        # Create image strip
        picture_all = Image.new("L", (56 * NUMBER, 56))
        for i, picture_unit in enumerate(w_picture):
            picture_all.paste(picture_unit, (i * 56, 0))
        # Set save path
        save_way = os.path.join(data_way, "gewichte.png")
        picture_all.convert("RGB").save(save_way)
        print(f"Saved {NUMBER} weights as {save_way}")

    def save_misclassified(self, net, X, y, max_picture,
                           data_way="falsch_klassifiziert"):
        '''
        Save misclassified test images as PNG files.

        Parameters
        ----------
        net : NeuralNetTraining
            The trained network.
        X : np.ndarray
            Test images
        y : np.ndarray
            True labels
        max_picture : int
            Maximum number of misclassified images to save.
        data_way : str
            Save path (filename prefix)
        '''
        wrong_classified = 0
        # Ensure folder exists
        os.makedirs(data_way, exist_ok=True)
        for i in range(len(X)):
            petition = X[i].flatten() / 255.0
            output_data = net.predict(petition)
            available = np.argmax(output_data)
            if available != y[i]:
                arr = X[i].astype(np.uint8)
                picture = Image.fromarray(arr)
                picture = picture.resize((115, 115), Image.Resampling.NEAREST)
                filename = f"{wrong_classified}_true_{y[i]}_pred_{available}.png"
                way = os.path.join(data_way, filename)
                try:
                    picture.save(way)
                except Exception as e:
                    print(f"Error while saving image: {e}")
                wrong_classified += 1
                if wrong_classified >= max_picture:
                    break
        if wrong_classified == 0:
            print("No misclassified images found.")
        else:
            print(f"Saved {wrong_classified} misclassified images.")

    def save_misclassified_table(self, net, X, y, max_picture,
                                 data_way="falsch_klassifiziert_tabelle"):
        '''
        Create a PNG table with true and predicted labels of misclassified
        test images.

        Parameters
        ----------
        net : NeuralNetTraining
            The trained network.
        X : np.ndarray
            Test images
        y : np.ndarray
            True labels
        max_picture : int
            Maximum number of misclassified images to list.
        data_way : str
            Filename for the output graphic.
        '''
        # Collect misclassified examples
        wrongs = []
        for i in range(len(X)):
            picture = X[i].flatten() / 255.0
            output_data = net.predict(picture)
            pred = np.argmax(output_data)
            if pred != y[i]:
                wrongs.append((i, int(y[i]), int(pred)))
                # Save image
                filename = f"{i}_true_{int(y[i])}_pred_{int(pred)}.png"
                way = os.path.join(data_way, filename)
                try:
                    os.makedirs(data_way, exist_ok=True)
                    arr_unit8 = (picture.reshape(28, 28) * 255).astype(np.uint8)
                    img_to_save = Image.fromarray(arr_unit8, mode="L")
                    img_to_save = img_to_save.resize((115, 115), Image.Resampling.NEAREST)
                    img_to_save.save(way)
                except Exception as e:
                    print(f"Error while saving image {filename}: {e}")
                if len(wrongs) >= max_picture:
                    break
        # Table size
        rows = len(wrongs) + 1
        cols = 3
        cell_w = 150
        cell_h = 40
        width = cols * cell_w
        height = rows * cell_h
        # Create table as image
        picture = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(picture)
        # Optional: font (works only if a TTF file is available)
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except:
            font = None  # Use default font
        # Header
        header = ["Image index", "True", "Prediction"]
        for c, text in enumerate(header):
            draw.rectangle([c*cell_w, 0, (c+1)*cell_w, cell_h], outline="black",
                           width=2)
            draw.text((c*cell_w+10, 5), text, fill="black", font=font)
        # Rows
        for r, (idx, label_true, label_pred) in enumerate(wrongs, start=1):
            for c, value in enumerate([idx, label_true, label_pred]):
                draw.rectangle([c*cell_w, r*cell_h, (c+1)*cell_w, (r+1)*cell_h],
                               outline="black", width=1)
                draw.text((c*cell_w+10, r*cell_h+5), str(value), fill="black",
                          font=font)
        # Save table
        table_way = os.path.join(data_way, "falsch_klassifiziert_tabelle.png")
        picture.save(table_way)
        print(f"Saved table of misclassified images as {table_way}")

    def save_error_heatmap(self, net, X, y, max_picture,
                           data_way="heatmap_fehler"):
        '''
        Create a heatmap (PNG) showing which image regions most often lead to
        misclassifications.

        Parameters
        ----------
        net : NeuralNetTraining
            The trained network.
        X : np.ndarray
            Test images
        y : np.ndarray
            True labels
        max_picture : int
            Maximum number of misclassified images to include.
        data_way : str
            Filename for the output graphic.
        '''
        # Heatmap array for errors
        error_map = np.zeros((28, 28), dtype=np.float32)
        count = 0
        for i in range(len(X)):
            x_img = X[i].reshape(28, 28)
            picture = X[i].flatten() / 255.0
            output_data = net.predict(picture)
            pred = np.argmax(output_data)
            if pred != y[i]:
                # Accumulate pixel intensities of misclassified images
                error_map += x_img
                count += 1
                if count >= max_picture:
                    break
        if count == 0:
            print("No misclassifications found for heatmap.")
            return
        # Compute mean and normalize
        heatmap = error_map / count
        heatmap = 255 * (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)
        heatmap_img = Image.fromarray(heatmap.astype(np.uint8)).convert("L").resize((140, 140), Image.Resampling.NEAREST)
        # Create colored heatmap (optional)
        heatmap_img = heatmap_img.convert("RGB")
        pixels = heatmap_img.load()
        for i in range(140):
            for j in range(140):
                # The brighter, the redder (e.g., for errors)
                value = heatmap_img.getpixel((i, j))[0]
                pixels[i, j] = (value, 0, 255 - value)  # red-blue gradient
        os.makedirs((data_way), exist_ok=True)
        save_way = os.path.join(data_way, "heatmap_fehler.png")
        heatmap_img.save(save_way)
        print(f"Saved heatmap of misclassifications as {save_way}")

    def save_error_heatmap_csv(self, net, X, y, max_picture,
                               data_way="heatmap_fehler.csv"):
        '''
        Save the error-pixel heatmap as a CSV file (values only, no header,
        semicolon-separated).

        Parameters
        ----------
        net : NeuralNetTraining
            The trained network.
        X : np.ndarray
            Test images (N, 28, 28 or N, 784).
        y : np.ndarray
            True labels (N).
        max_picture : int
            Max. number of misclassifications to include.
        data_way : str
            Path to the output file (.csv).
        '''
        error_map = np.zeros((28, 28), dtype=np.float32)
        count = 0
        for i in range(len(X)):
            x_img = X[i].reshape(28, 28)
            picture = X[i].flatten() / 255.0
            output_data = net.predict(picture)
            pred = np.argmax(output_data)
            if pred != y[i]:
                error_map += x_img
                count += 1
                if count >= max_picture:
                    break
        if count == 0:
            print("No misclassifications found for heatmap.")
            return
        heatmap = error_map / count
        # Create folder for the CSV file if necessary
        dir_name = os.path.dirname(data_way)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # Write as CSV text file
        with open(data_way, "w", encoding="utf-8") as f:
            for row in heatmap:
                f.write(";".join(f"{v:.2f}" for v in row) + "\n")
        print(f"Saved error heatmap as text: {data_way}")

    def save_confusion_heatmap(self, net, X, y, data_way="confusion_heatmap"):
        '''
        Create and save a heatmap of the confusion matrix as a PNG image.

        Parameters
        ----------
        net : NeuralNetTraining
            The trained network.
        X : np.ndarray
            Test images
        y : np.ndarray
            True labels
        data_way : str
            Filename for the output graphic.
        '''
        preds = np.argmax(net.predict(X.reshape(-1, net.input_nodes) / 255.0),
                          axis=1)
        num_classes = np.max(y) + 1
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y, preds):
            matrix[int(true), int(pred)] += 1
        max_val = np.max(matrix)
        cell = 42
        picture = Image.new("RGB", (cell * num_classes, cell * num_classes), "white")
        draw = ImageDraw.Draw(picture)
        for r in range(num_classes):
            for c in range(num_classes):
                val = matrix[r, c]
                # Simple heatmap from white (0) to red (max)
                rot = int(255 * val / max_val) if max_val > 0 else 0
                col = (255, 255 - rot, 255 - rot)
                draw.rectangle([c*cell, r*cell, (c+1)*cell, (r+1)*cell], fill=col)
                draw.text((c*cell+cell//4, r*cell+cell//4), str(val), fill="black")
        # Ensure folder exists
        os.makedirs(data_way, exist_ok=True)
        # Path with filename
        save_way = os.path.join(data_way, "confusion_heatmap.png")
        picture.save(save_way)
        print(f"Saved heatmap of the confusion matrix: {save_way}")

    def save_confusion_matrix(self, net, X, y, data_way="confusion_matrix.png"):
        '''
        Create and save a confusion matrix as a PNG image.

        Parameters
        ----------
        net : NeuralNetTraining
            The trained network.
        X : np.ndarray
            Test images
        y : np.ndarray
            True labels
        data_way : str
            Filename for the output graphic.
        '''
        # Compute predictions
        preds = np.argmax(net.predict(X.reshape(-1, net.input_nodes) / 255.0), axis=1)
        num_classes = np.max(y) + 1
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y, preds):
            matrix[int(true), int(pred)] += 1
        # Image size
        cell_w = 55
        cell_h = 40
        width = (num_classes + 1) * cell_w
        height = (num_classes + 1) * cell_h
        picture = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(picture)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = None
        # Header
        for i in range(num_classes):
            draw.rectangle([cell_w + i*cell_w, 0, cell_w + (i+1)*cell_w, cell_h],
                           outline="black", width=2)
            draw.text((cell_w + i*cell_w + 10, 8), str(i), fill="black", font=font)
            draw.rectangle([0, cell_h + i*cell_h, cell_w, cell_h + (i+1)*cell_h],
                           outline="black", width=2)
            draw.text((10, cell_h + i*cell_h + 8), str(i), fill="black", font=font)
        # Fill values
        for r in range(num_classes):
            for c in range(num_classes):
                val = matrix[r, c]
                col = "red" if (r == c) else "black"
                draw.rectangle([cell_w + c*cell_w, cell_h + r*cell_h,
                                cell_w + (c+1)*cell_w, cell_h + (r+1)*cell_h],
                               outline="black", width=1)
                draw.text((cell_w + c*cell_w + 8, cell_h + r*cell_h + 10),
                          str(val), fill=col, font=font)
        # Labels
        draw.text((cell_w//3, 2), "true", fill="black", font=font)
        draw.text((cell_w*2, height - cell_h + 2), "predicted →", fill="black",
                  font=font)
        picture.save(data_way)
        print(f"Saved confusion matrix as table: {data_way}")

    def save_confusion_matrix_csv(self, net, X, y,
                                  data_way="confusion_matrix.csv"):
        '''
        Save the confusion matrix as a CSV file (values only, with header,
        semicolon-separated).

        Parameters
        ----------
        net : NeuralNetTraining
            The trained network.
        X : np.ndarray
            Test images.
        y : np.ndarray
            True labels.
        data_way : str
            Path to the output file (.txt).
        '''
        # Compute confusion matrix
        preds = np.argmax(net.predict(X.reshape(-1, net.input_nodes) / 255.0), axis=1)
        num_classes = np.max(y) + 1
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y, preds):
            matrix[int(true), int(pred)] += 1
        # Create folder for the CSV file if necessary
        dir_name = os.path.dirname(data_way)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # Write as CSV text file
        with open(data_way, "w", encoding="utf-8") as f:
            # Header e.g., for export to Excel
            f.write("true/predicted;" + ";".join(str(i) for i in range(num_classes)) + "\n")
            for r in range(num_classes):
                row = ";".join(str(matrix[r, c]) for c in range(num_classes))
                f.write(f"{r};{row}\n")
        print(f"Saved confusion matrix as text: {data_way}")

    def save_sample_picture_train_labeled(self, data_way="mnist_bsp_train_labeled"):
        '''
        Save training samples with an embedded label at the top-left corner.

        Parameters
        ----------
        data_way : str
            Directory prefix for the output files.
        '''
        # Create folder
        os.makedirs(data_way, exist_ok=True)
        for i in range(NUMBER):
            picture = Image.fromarray(self.X_train[i].astype(np.uint8))
            picture_bigger = picture.resize((115, 115), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(picture_bigger)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = None
            # Overlay label at top left
            label = str(self.y_train[i])
            draw.rectangle([0, 0, 35, 35], fill="yellow")
            draw.text((8, 2), label, fill="black", font=font)
            # Folder for saving
            filename = f"trainbild_{i}_label_{self.y_train[i]}.png"
            way = os.path.join(data_way, filename)
            picture_bigger.save(way)
        print(f"Saved {NUMBER} training samples with label overlay.")

    def save_sample_picture_test_labeled(self, data_way="mnist_bsp_test_labeled"):
        '''
        Save test samples with an embedded label at the top-left corner.

        Parameters
        ----------
        data_way : str
            Directory prefix for the output files.
        '''
        # Create folder
        os.makedirs(data_way, exist_ok=True)
        for i in range(NUMBER):
            picture = Image.fromarray(self.X_test[i].astype(np.uint8))
            picture_bigger = picture.resize((115, 115), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(picture_bigger)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = None
            label = str(self.y_test[i])
            draw.rectangle([0, 0, 35, 35], fill="yellow")
            draw.text((8, 2), label, fill="black", font=font)
            # Folder for saving
            filename = f"testbild_{i}_label_{self.y_test[i]}.png"
            way = os.path.join(data_way, filename)
            picture_bigger.save(way)
        print(f"Saved {NUMBER} test samples with label overlay.")


# Main program
if __name__ == "__main__":
    # Timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Load and split data
    loader = MNISTDownLoad(test_size=0.2)
    loader.load_and_split_mnist_data()
    loader.save_mnist_data_new(os.path.join(OUTPUT_DIR,
                                            f"mnist_daten_split_{timestamp}.npz"))
    # Initialize network
    net = NeuralNetTraining()
    # Start time for measuring training time
    start_time = time.time()
    # Training with loss and accuracy lists
    error_list, acc_list = net.training_epochs(loader.X_train, loader.y_train,
                                               loader.X_test, loader.y_test)
    # End time for measuring training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    # Inference
    petition = loader.X_test[0].flatten() / 255.0
    output_data = net.predict(petition.reshape(1, -1))
    digit = np.argmax(output_data)
    print("Network output:", output_data.flatten())
    print("Predicted digit:", digit)
    # Visualization
    vis = Visualizer(loader.X_train, loader.y_train, loader.X_test,
                     loader.y_test)
    vis.save_sample_picture_train(data_way=os.path.join(OUTPUT_DIR,
                                                        f"mnist_bsp_train_{timestamp}"))
    vis.save_sample_picture_test(data_way=os.path.join(OUTPUT_DIR,
                                                       f"mnist_bsp_test_{timestamp}"))
    vis.save_error_curve(error_list, data_way=os.path.join(OUTPUT_DIR,
                                                           f"fehlerkurve_{timestamp}.png"))
    vis.save_accuracy(acc_list, data_way=os.path.join(OUTPUT_DIR,
                                                      f"genauigkeitskurve_{timestamp}.png"))
    vis.save_train_vs_test_curve(error_list, acc_list,
                                 data_way=os.path.join(OUTPUT_DIR, f"train_vs_test_{timestamp}.png"))
    vis.save_weights(net.weight_input_hidden, NUMBER,
                     data_way=os.path.join(OUTPUT_DIR, f"gewichte_{timestamp}.png"))
    vis.save_misclassified(net, loader.X_test, loader.y_test, max_picture=10,
                           data_way=os.path.join(OUTPUT_DIR, f"falsch_klassifiziert_{timestamp}"))
    vis.save_misclassified_table(net, loader.X_test, loader.y_test, max_picture=10,
                                 data_way=os.path.join(OUTPUT_DIR, f"falsch_klassifiziert_tabelle_"
                                                           f"{timestamp}.png"))
    vis.save_error_heatmap(net, loader.X_test, loader.y_test, max_picture=100,
                           data_way=os.path.join(OUTPUT_DIR, f"heatmap_fehler_{timestamp}.png"))
    vis.save_error_heatmap_csv(net, loader.X_test, loader.y_test, max_picture=100,
                               data_way=os.path.join(OUTPUT_DIR, f"heatmap_fehler_{timestamp}.csv"))
    vis.save_confusion_matrix(net, loader.X_test, loader.y_test,
                              data_way=os.path.join(OUTPUT_DIR, f"confusion_matrix_{timestamp}.png"))
    vis.save_confusion_heatmap(net, loader.X_test, loader.y_test,
                               data_way=os.path.join(OUTPUT_DIR, f"confusion_heatmap_{timestamp}.png"))
    vis.save_confusion_matrix_csv(net, loader.X_test, loader.y_test,
                                  data_way=os.path.join(OUTPUT_DIR, f"confusion_matrix_{timestamp}.csv"))
    vis.save_sample_picture_train_labeled(data_way=os.path.join(OUTPUT_DIR,
                                            f"mnist_bsp_train_labeled_{timestamp}"))
    vis.save_sample_picture_test_labeled(data_way=os.path.join(OUTPUT_DIR,
                                            f"mnist_bsp_test_labeled_{timestamp}"))

    # Project results block
    with open(os.path.join(OUTPUT_DIR, f"projekt_reflexion_{timestamp}.txt"),
              "w", encoding="utf-8") as f:
        f.write("Project completion – Reflection and evaluation\n")
        f.write("===========================================\n\n")
        f.write("╔══════════════════════════════════════════════════════════════════╗\n")
        f.write("║                 NEURAL NETWORK: PARAMETER TABLE                  ║\n")
        f.write("╠════════════════════════════════════════╦═════════════════════════╣\n")
        f.write(f"║ Input nodes                            ║ {INPUT_NODES:<20}    ║\n")
        f.write(f"║ Hidden nodes                           ║ {HIDDEN_NODES:<20}    ║\n")
        f.write(f"║ Output nodes                           ║ {OUTPUT_NODES:<20}    ║\n")
        f.write(f"║ Learning rate                          ║ {LEARNING_RATE:<20}    ║\n")
        f.write(f"║ Epochs                                 ║ {EPOCHS:<20}    ║\n")
        f.write(f"║ Batch size                             ║ {BATCH_SIZE:<20}    ║\n")
        f.write(f"║ Training samples                       ║ {len(loader.X_train):<20}    ║\n")
        f.write(f"║ Test samples                           ║ {len(loader.X_test):<20}    ║\n")
        f.write(f"║ Loss (last epoch)                      ║ {error_list[-1]:<20.6f}    ║\n")
        if acc_list[-1] is not None:
            f.write(f"║ Accuracy (last epoch)                  ║ {acc_list[-1]*100:<17.2f} %     ║\n")
        else:
            f.write(f"║ Accuracy (last epoch)                  ║ Not computed           ║\n")
        # → Add training time:
        f.write(f"║ Training time                          ║ {training_time/60:<16.2f} min    ║\n")
        f.write(f"║ Weight initialization                  ║ uniform(-0.5, 0.5)      ║\n")
        f.write(f"║ Bias used                              ║ Yes                     ║\n")
        f.write(f"║ Activation function                    ║ Sigmoid                 ║\n")
        f.write(f"║ Loss function                          ║ MSE (Mean Squared Error)║\n")
        f.write(f"║ Optimizer                              ║ SGD (Gradient Descent)  ║\n")
        f.write(f"║ Random seed                            ║ 42                      ║\n")
        f.write(f"║ Training start (timestamp)             ║ {timestamp:<20}    ║\n")
        f.write("╚════════════════════════════════════════╩═════════════════════════╝\n")
        f.write("\n")
    # Final message
    print(f"All results, images and documentation were saved in the folder "
          f"{OUTPUT_DIR} with timestamp {timestamp}.")
