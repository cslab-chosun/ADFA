# Abstraction and Decision Fusion Architecture for Resource-Aware Image Understanding

This repository contains the deployment code for the paper titled "Abstraction and Decision Fusion Architecture for Resource-Aware Image Understanding with Application on Handwriting Character Classification."

## Installation

Before running the application, ensure you have installed all the required packages listed in `requirements.txt`. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Download and Import Datasets**: All datasets will be downloaded and imported from the `keras.datasets` library. Please ensure to run the requirements first.

2. **Set Parameters**: At the beginning of running the application, set the `mod_problem` and `mode_type` parameters in the `main_anfis.py` file:
   - For training on the MNIST dataset, set the `mod_problem` parameter to `"mnist"`.
   - For the EMNIST dataset, set the `mod_problem` parameter to `"emnist"`. Additionally, if selecting the EMNIST dataset, ensure to set the `mode_type` parameter accordingly:
     - For EMNIST letters, set `mode_type` to `"letter"`.
     - For EMNIST balanced, set `mode_type` to `"balanced"`.

3. **Run the Application**: Execute the following command to run the application:

```bash
python main_anfis.py
```

4. **Training the ANFIS Model**: After the initial setup, use the results from the previous section to train the ANFIS model. Run the following command:
   - For the MNIST application:

```bash
python test.py mnist
```

   - For the EMNIST application, ensure to modify the `data_file` parameter in `test.py`, then run the following command:

```bash
python test.py emnist
```

## License

This project is licensed under the CSLAB_Chosun License.