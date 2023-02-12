
# import required libraries
class svm_test():
    def svm_pred(Xt,model_path):
        import pickle
        loaded_model = pickle.load(open(model_path, 'rb'))#"model_svm_second"
        y_pred1 = loaded_model.predict_proba(Xt)
        return(y_pred1)

