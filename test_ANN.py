class ann_test():
    def ann_pred(Xt,model_path):
        from tensorflow import keras
        import numpy as np

        model=keras.models.load_model(model_path)#'model_Ann_4*28'
        y_pred = model.predict(Xt)
        return y_pred