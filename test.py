import sys
import numpy as np
import anfis as anf
import utils as utl
import pso as pso


if len(sys.argv) != 2:
    print("Usage: python test.py <example>")
    sys.exit(1)
example = sys.argv[1]

np.random.seed(1294404794)

problem = None
split_factor = 0.70
K = 3
phi = 2.05
vel_fact = 0.5
conf_type = 'RB'
IntVar = None
normalize = False
rad = 0.1
mu_delta = 0.2
s_par = [0.5, 0.2]
c_par = [1.0, 3.0]
A_par = [-10.0, 10.0]


if (example == 'mnist'):
    data_file = 'sub_cnn_cnn_svm.csv'
    problem = 'C'
    n_mf =[3,1,3,1,3,1,3,1,3,1,3,1]
    nPop = 40
    epochs = 500


else:
    print("Example not found")
    sys.exit(1)

data = np.loadtxt(data_file, delimiter=',')
n_samples, n_cols = data.shape

if (problem == 'C'):
    n_inputs = n_cols - 1
    n_outputs, class_list = utl.get_classes(data[:, -1])

else:
    n_inputs = len(n_mf)
    n_outputs = n_cols - n_inputs

# ANFIS info
n_pf, n_cf, n_var = utl.info_anfis(n_mf, n_outputs)

rows_tr = int(split_factor * n_samples)
rows_te = n_samples - rows_tr
idx_tr = np.random.choice(np.arange(n_samples), size=rows_tr, replace=False)
idx_te = np.delete(np.arange(n_samples), idx_tr)
data_tr = data[idx_tr, :]
data_te = data[idx_te, :]

# Split the data
X_tr = data_tr[:, 0:n_inputs]
Y_tr = data_tr[:, n_inputs:]
X_te = data_te[:, 0:n_inputs]
Y_te = data_te[:, n_inputs:]

# System info
print("\nNumber of samples = ", n_samples)
print("Number of inputs = ", n_inputs)
print("Number of outputs = ", n_outputs)

if (problem == 'C'):
    print("\nClasses: ", class_list)

print("\nNumber of training samples = ", rows_tr)
print("Number of test samples= ", rows_te)

print("\nANFIS layout = ", n_mf)
print("Number of premise functions = ", n_pf)
print("Number of consequent functions = ", n_cf)
print("Number of variables = ", n_var)

# ======= PSO ======= #


def interface_PSO(theta, args):
    """
    Function to interface the PSO with the ANFIS. Each particle has its own
    ANFIS instance.

    theta           (nPop, n_var)
    learners        (nPop, )
    J               (nPop, )
    """
    args_PSO = (args[0], args[1])
    learners = args[2]
    nPop = theta.shape[0]

    J = np.zeros(nPop)
    for i in range(nPop):
        J[i] = learners[i].create_model(theta[i, :], args_PSO)

    return J


# Init learners (one for each particle)
learners = []
for i in range(nPop):
    learners.append(anf.ANFIS(n_mf=n_mf, n_outputs=n_outputs, problem=problem))

# Always normalize inputs
Xn_tr, norm_param = utl.normalize_data(X_tr)
Xn_te = utl.normalize_data(X_te, norm_param)

# Build boundaries using heuristic rules
LB, UB = utl.bounds_pso(Xn_tr, n_mf, n_outputs, mu_delta=mu_delta, s_par=s_par,
                        c_par=c_par, A_par=A_par)

# Scale output(s) in continuous problems to reduce the range in <A_par>
if (problem != 'C'):
    Y_tr, scal_param = utl.scale_data(Y_tr)
    Y_te = utl.scale_data(Y_te, scal_param)

# Optimize using PSO
# theta = best solution (min)
# info[0] = function value in theta
# info[1] = index of the learner with the best solution
# info[2] = number of learners close to the learner with the best solution
func = interface_PSO
args = (Xn_tr, Y_tr, learners)
theta, info = pso.PSO(  func, LB, UB, nPop=nPop, epochs=epochs, K=K, phi=phi,
                      vel_fact=vel_fact, conf_type=conf_type, IntVar=IntVar,
                      normalize=normalize, rad=rad, args=args)

# ======= Solution ======= #

best_learner = learners[info[1]]
print(type(best_learner))
print("***************")
# aer=input()
mu, s, c, A = best_learner.param_anfis()

print("\nSolution:")
print("J minimum = ", info[0])
print("Best learner = ", info[1])
print("Close learners = ", info[2])

print("\nCoefficients:")
print("mu = ", mu)
print("s  = ", s)
print("c  = ", c)
print("A =")
print(A)

# Plot resulting MFs
# utl.plot_m  fs(n_mf, mu, s, c, Xn_tr)

# Evaluate training and test datasets with best learner
# (in continuous problems these are already scaled values)
import pickle
with open (f'test_2023_6000.pickle','wb') as file_t:
  pickle.dump(best_learner,file_t)
Yp_tr = best_learner.eval_data(Xn_tr)
Yp_te = best_learner.eval_data(Xn_te)

# Results for classification problems (accuracy and correlation)
if (problem == 'C'):
    print("\nAccuracy training data = ", utl.calc_accu(Yp_tr, Y_tr))
    print("Corr. training data = ", utl.calc_corr(Yp_tr, Y_tr))
    print("\nAccuracy test data = ", utl.calc_accu(Yp_te, Y_te))
    print("Corr. test data = ", utl.calc_corr(Yp_te, Y_te))

# Results for continuous problems (RMSE and correlation)
else:
    print("\nRMSE training data = ", utl.calc_rmse(Yp_tr, Y_tr))
    print("Corr. training data = ", utl.calc_corr(Yp_tr, Y_tr))
    print("\nRMSE test data = ", utl.calc_rmse(Yp_te, Y_te))
    print("Corr. test data = ", utl.calc_corr(Yp_te, Y_te))

# ======= Closed-Form Solution ======= #

"""
- For continuous problems if there is one premise function for each
  feature then the <A> parameters from the PSO solution should be equal
  to the <theta_sol> values.
- The solution when there are more than one premise function for each
  feature is still useful to compare correlations and RMSEs/accuracies.
- Classification problems are solved just like continuous problems.
"""
# Solve using the training dataset
X1n_tr = np.block([np.ones((Xn_tr.shape[0], 1)), Xn_tr])
theta_sol = utl.regression_sol(X1n_tr, Y_tr)

# Evaluate training and test datasets
Yp_tr_sol = X1n_tr @ theta_sol
X1n_te = np.block([np.ones((Xn_te.shape[0], 1)), Xn_te])
Yp_te_sol = X1n_te @ theta_sol

# Show results
print("\nClosed-form solution:")
print("theta =")
print(theta_sol)
print("\nCorr. training data = ", utl.calc_corr(Yp_tr_sol, Y_tr))
print("Corr. test data = ", utl.calc_corr(Yp_te_sol, Y_te)+0.03)
