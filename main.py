import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data import FEATURES, penguinData
from experiment import ExperimentRunner
from models import SIGMOID, TANH
#TODO: @OLA,@Muram -> practice: rewrite full code from memory to understand it
#TODO: @Taj,@Khalid -> Solve Task 2

#TODO: @Fahim -> Do the integration work

#Message from Taj: work on your own local branches. No 'Accidental pushes.' like last time :) whatever you want to do, do on your own machine but keep straight to tthe goal. 


def draw_confusion(ax, result):
    """Plot actual classes by row and predicted classes by column."""
    matrix = result.conf
    
    ax.imshow(matrix, cmap="Blues", vmin=0)
    
    ax.set_xticks(range(len(result.classNames)), result.classNames, rotation=20, ha="right")
    
    ax.set_yticks(range(len(result.classNames)), result.classNames)
    
    ax.set_xlabel("Predicted class")
    
    ax.set_ylabel("Actual class")
    
    for row in range(len(result.classNames)):
    
        for column in range(len(result.classNames)):
    
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
    
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    
    ax.set_title(f"Test confusion matrix - accuracy {result.Ts_acc * 100:.1f}%")


class App:
    def __init__(self, root, data):
        # Keep the Task 1 layout: controls on the left and results on the right.
        self.root, self.data, self.result = root, data, None
        root.title("Lab 4 - Back-Propagation Classifier")
        root.geometry("1150x650")

        left = ttk.Frame(root, padding=10)
        left.pack(side="left", fill="y")

        self.layers = tk.StringVar(value="1")
        self.neurons = tk.StringVar(value="12")
        self.eta = tk.StringVar(value="0.1")
        self.epochs = tk.StringVar(value="500")
        self.activation = tk.StringVar(value=SIGMOID)
        self.bias = tk.BooleanVar(value=True)

        fields = [
            ("Hidden layers", ttk.Entry(left, textvariable=self.layers)),
            ("Neurons per layer", ttk.Entry(left, textvariable=self.neurons)),
            ("Learning rate (eta)", ttk.Entry(left, textvariable=self.eta)),
            ("Epochs (m)", ttk.Entry(left, textvariable=self.epochs)),
        ]
        for row, (label, widget) in enumerate(fields):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
            widget.grid(row=row, column=1, sticky="ew", pady=2)

        ttk.Checkbutton(left, text="Add bias", variable=self.bias).grid(
            row=4, column=0, columnspan=2, sticky="w")
        for row, activation in enumerate((SIGMOID, TANH), start=5):
            ttk.Radiobutton(left, text=activation, value=activation, variable=self.activation).grid(
                row=row, column=0, columnspan=2, sticky="w")

        ttk.Button(left, text="Train && Test", command=self.on_train).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=8)
        self.info = ttk.Label(left, text="", justify="left", wraplength=310)
        self.info.grid(row=8, column=0, columnspan=2, sticky="w")

        ttk.Separator(left).grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(left, text="Classify one sample (after training)").grid(
            row=10, column=0, columnspan=2, sticky="w")
        self.sample_inputs = {}
        for row, feature in enumerate(FEATURES, start=11):
            hint = f" ({' / '.join(data.origins)})" if feature == "OriginLocation" else ""
            ttk.Label(left, text=feature + hint).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
            if feature == "OriginLocation":
                widget = ttk.Combobox(left, values=data.origins, state="readonly")
            else:
                widget = ttk.Entry(left)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.sample_inputs[feature] = widget

        ttk.Button(left, text="Classify", command=self.on_classify).grid(
            row=16, column=0, columnspan=2, sticky="ew", pady=6)
        self.sample_result = ttk.Label(left, text="Predicted species: -")
        self.sample_result.grid(row=17, column=0, columnspan=2, sticky="w")

        self.figure = Figure(figsize=(8, 4.5), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().configure(width=400, height=300)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def parse_network(self):
        """Read and validate the hidden-layer settings."""
        try:
            layer_count = int(self.layers.get().strip())
            hidden_layers = [int(value.strip()) for value in self.neurons.get().split(",") if value.strip()]
            eta = float(self.eta.get().strip())
            epochs = int(self.epochs.get().strip())
        except ValueError:
            raise ValueError("Enter whole numbers for layers, neurons, and epochs, and a numeric learning rate.") from None
        if layer_count < 1:
            raise ValueError("Enter at least one hidden layer.")
        if len(hidden_layers) != layer_count or any(size < 1 for size in hidden_layers):
            raise ValueError("Enter one positive neuron count per hidden layer, separated by commas.")
        if eta <= 0 or epochs < 1:
            raise ValueError("Learning rate must be positive and epochs must be at least 1.")
        return hidden_layers, eta, epochs

    def on_train(self):
        """Train the network, then refresh the confusion matrix and summary."""
        try:
            hidden_layers, eta, epochs = self.parse_network()
            self.result = ExperimentRunner(
                self.data, hidden_layers, eta, epochs, self.bias.get(), self.activation.get())
        except (ValueError, IndexError) as error:
            messagebox.showerror("Invalid input", str(error))
            return

        self.figure.clear()
        draw_confusion(self.figure.add_subplot(1, 1, 1), self.result)
        self.canvas.draw()
        self.info.configure(
            text=f"Train accuracy: {self.result.Tr_acc * 100:.1f}%\n"
                 f"Test accuracy: {self.result.Ts_acc * 100:.1f}%")
        self.sample_result.configure(text="Predicted species: -")

    def on_classify(self):
        """Classify the entered five-feature sample with the trained network."""
        if self.result is None:
            messagebox.showinfo("Classify", "Train the network first.")
            return
        values = [self.sample_inputs[feature].get().strip() for feature in FEATURES]
        try:
            predicted = self.result.classifierFn(values)
        except ValueError as error:
            messagebox.showerror("Cannot classify", str(error))
            return
        self.sample_result.configure(text=f"Predicted species: {predicted}")


def main():
    # Load the bird data, then start the application window.
    root = tk.Tk()
    try:
        data = penguinData()
    except (OSError, KeyError, ValueError) as error:
        messagebox.showerror("Cannot load dataset", str(error))
        root.destroy()
        return
    App(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
