from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
  
def result_plots(z_mean,z_log_var,z_mean_t,z_log_var_t,k_train,k_test,save_path_train = None,save_path_test = None):

    mser = mean_squared_error(k_train, z_mean)
    r2 = r2_score(k_train, z_mean)
    mser_t = mean_squared_error(k_test, z_mean_t)
    r2_t = r2_score(k_test, z_mean_t)
    
    #upper and lower bounds to calculate 95% interval
    z_std = np.sqrt(np.exp(z_log_var))  # Standard deviation from log variance
    z_lower = z_mean -  1.96 *z_std  # Lower bound
    z_upper = z_mean +  1.96* z_std  # Upper bound
    z_lower = z_lower.reshape(-1)
    z_upper = z_upper.reshape(-1)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # Plot ground truth values
    ax1.scatter(k_train, k_train, color='blue', label='Ground Truth')
    # Plot predicted values
    ax1.scatter(k_train, z_mean, color='red',alpha = 0.7,  label='Predicted')
    # Plot the confidence intervals as vertical lines
    k_train = k_train.reset_index(drop=True)
    for i in range(len(z_mean)):
        plt.plot([k_train[i], k_train[i]], [z_lower[i], z_upper[i]], color='gray',alpha=0.4)

    # Set labels and make axis equal
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    ax1.axis('equal')
    # Add a legend to differentiate between the two sets of points
    plt.legend(loc ='lower right')
    plt.title('Predictions on Train Data')
    if save_path_train:
        plt.savefig(save_path_train)
    else:
        plt.show()


    #plots for test predictions
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    z_std_t = np.sqrt(np.exp(z_log_var_t))  # Standard deviation from log variance
    z_lower_t = z_mean_t - 1.96 * z_std_t  # Lower bound
    z_upper_t = z_mean_t + 1.96 * z_std_t  # Upper bound
    z_lower_t = z_lower_t.reshape(-1)
    z_upper_t = z_upper_t.reshape(-1)
    # Plot ground truth values
    ax2.scatter(k_test, k_test, color='blue', label='Ground Truth')
    # Plot predicted values
    ax2.scatter(k_test, z_mean_t, color='red',alpha = 0.7,  label='Predicted')
    k_test = k_test.reset_index(drop=True)
    # Plot the confidence intervals as vertical lines
    for i in range(len(z_mean_t)):
        plt.plot([k_test[i], k_test[i]], [z_lower_t[i], z_upper_t[i]], color='gray',alpha=0.4)


    # Set labels and make axis equal
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    ax2.axis('equal')
    # Add a legend to differentiate between the two sets of points
    plt.legend(loc ='lower right')
    plt.title('Predictions on Test Data')
    if save_path_test:
        plt.savefig(save_path_test)
    else:
        plt.show()

    return mser, r2, mser_t, r2_t



def plot(history,save_path = None):

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()