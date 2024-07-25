import numpy as np

def regress():
    # A = np.array([
    #     [1, 1],
    #     [1, 2],
    #     [1, 3],

    # ])

    # b = np.array([
    #     1,2,2
    # ])
    all = np.genfromtxt("C:\\Users\\rejil\\OneDrive\\Documents\\GitHub\\Personal-CLI\\Mobile-Price-Prediction-cleaned_data.csv", skip_header=1, delimiter=",")

    A = np.hstack((all[:, :-1], np.ones((all.shape[0], 1))))
    b = all[:, -1:]
    # print()
    weights = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)
    print(weights.shape)
    return weights
    # print(b.shape)

if __name__ == "__main__":
    print(regress())
