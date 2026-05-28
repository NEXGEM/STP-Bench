import pycpd
from builtins import super
import numbers
import numpy as np
import cv2

class EMRegistration(object):
    """
    Expectation maximization point cloud registration.
    Adapted from Pure Numpy Implementation of the Coherent Point Drift Algorithm: 
    https://github.com/siavashk/pycpd


    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    TY: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    """

    def __init__(self, X, Y, sigma2=None, max_iterations=None, tolerance=None, w=None, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

        self.X = X
        self.Y = Y
        self.TY = Y
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N, ))
        self.P1 = np.zeros((self.M, ))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.

        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.
        
        Returns
        -------
        self.TY: numpy array
            MxD array of transformed source points.
        
        registration_parameters:
            Returned params dependent on registration method used. 
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :])**2, axis=2) # (M, N)
        P = np.exp(-P/(2*self.sigma2))
        c = (2*np.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

        den = np.sum(P, axis = 0, keepdims = True) # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + c

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()


class DeformableRegistration(EMRegistration):
    """
    Deformable registration.
    Adapted from Pure Numpy Implementation of the Coherent Point Drift Algorithm: 
    https://github.com/siavashk/pycpd

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.
    
    low_rank: bool
        Whether to use low rank approximation.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(self, alpha=None, beta=None, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))

        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
            self.E = 0.

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.dot(np.diag(self.P1), self.Y)
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.Y)

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                

        """
        self.W[:,2:]=0
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + np.dot(G, self.W)
        else:
            if self.low_rank is False:
                self.TY = self.Y + np.dot(self.G, self.W)

            elif self.low_rank is True:
                self.TY = self.Y + np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))
                return


    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = np.inf

        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(np.transpose(self.P1),  np.sum(
            np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.


        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W



def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).

    param
    ----------
    X: numpy array
        NxD array of points for target.
    
    Y: numpy array
        MxD array of points for source.
    
    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)
    


def gaussian_kernel(X, beta, Y=None):
    """
    Computes a Gaussian (RBF) kernel matrix between two sets of vectors.

    :param X: A numpy array of shape (n_samples_X, n_features) representing the first set of vectors.
    :param beta: The standard deviation parameter for the Gaussian kernel. It controls the spread of the kernel.
    :param Y: An optional numpy array of shape (n_samples_Y, n_features) representing the second set of vectors.
              If None, the function computes the kernel between `X` and itself (i.e., the Gram matrix).
    :return: A numpy array of shape (n_samples_X, n_samples_Y) representing the Gaussian kernel matrix.
             Each element (i, j) in the matrix is computed as:
             `exp(-||X[i] - Y[j]||^2 / (2 * beta^2))`
    """
    
    # If Y is not provided, use X for both sets, computing the kernel matrix between X and itself
    if Y is None:
        Y = X
    
    # Compute the difference tensor between each pair of vectors in X and Y
    # The resulting shape is (n_samples_X, n_samples_Y, n_features)
    diff = X[:, None, :] - Y[None, :, :]
    
    # Square the differences element-wise
    diff = np.square(diff)
    
    # Sum the squared differences across the feature dimension (axis 2) to get squared Euclidean distances
    # The resulting shape is (n_samples_X, n_samples_Y)
    diff = np.sum(diff, axis=2)
    
    # Apply the Gaussian (RBF) kernel formula: exp(-||X[i] - Y[j]||^2 / (2 * beta^2))
    kernel_matrix = np.exp(-diff / (2 * beta**2))
    
    return kernel_matrix



def low_rank_eigen(G, num_eig):
    """
    Calculate the top `num_eig` eigenvectors and eigenvalues of a given Gaussian matrix G.
    This function is useful for dimensionality reduction or when a low-rank approximation is needed.

    :param G: A square matrix (numpy array) for which the eigen decomposition is to be performed.
    :param num_eig: The number of top eigenvectors and eigenvalues to return, based on the magnitude of eigenvalues.
    :return: A tuple containing:
             - Q: A numpy array with shape (n, num_eig) containing the top `num_eig` eigenvectors of the matrix `G`. 
               Each column in `Q` corresponds to an eigenvector.
             - S: A numpy array of shape (num_eig,) containing the top `num_eig` eigenvalues of the matrix `G`.

    """
    
    # Perform eigen decomposition on matrix G
    # `S` will contain all the eigenvalues, and `Q` will contain the corresponding eigenvectors
    S, Q = np.linalg.eigh(G)
    
    # Sort eigenvalues in descending order based on their absolute values
    # Get the indices of the top `num_eig` largest eigenvalues
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    
    # Select the corresponding top eigenvectors based on the sorted indices
    Q = Q[:, eig_indices]  # Q now contains the top `num_eig` eigenvectors
    
    # Select the top `num_eig` eigenvalues based on the sorted indices
    S = S[eig_indices]  # S now contains the top `num_eig` eigenvalues
    
    return Q, S
    


def find_homography_translation_rotation(src_points, dst_points):
    """
    Find the homography between two sets of coordinates with only translation and rotation.

    :param src_points: A numpy array of shape (n, 2) containing source coordinates.
    :param dst_points: A numpy array of shape (n, 2) containing destination coordinates.
    :return: A 3x3 homography matrix.
    """
    # Ensure the points are in the correct shape
    assert src_points.shape == dst_points.shape
    assert src_points.shape[1] == 2

    # Calculate the centroids of the point sets
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Center the points around the centroids
    centered_src_points = src_points - src_centroid
    centered_dst_points = dst_points - dst_centroid

    # Calculate the covariance matrix
    H = np.dot(centered_src_points.T, centered_dst_points)

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Calculate the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Calculate the translation vector
    t = dst_centroid - np.dot(R, src_centroid)

    # Construct the homography matrix
    homography_matrix = np.eye(3)
    homography_matrix[0:2, 0:2] = R
    homography_matrix[0:2, 2] = t

    return homography_matrix



def apply_homography(coordinates, H):
    """
    Apply a 3x3 homography matrix to 2D coordinates.

    :param coordinates: A numpy array of shape (n, 2) containing 2D coordinates.
    :param H: A numpy array of shape (3, 3) representing the homography matrix.
    :return: A numpy array of shape (n, 2) with transformed coordinates.
    """
    # Convert (x, y) to homogeneous coordinates (x, y, 1)
    n = coordinates.shape[0]
    homogeneous_coords = np.hstack((coordinates, np.ones((n, 1))))

    # Apply the homography matrix
    transformed_homogeneous = np.dot(homogeneous_coords, H.T)

    # Convert back from homogeneous coordinates (x', y', w') to (x'/w', y'/w')
    transformed_coords = transformed_homogeneous[:, :2] / transformed_homogeneous[:, [2]]

    return transformed_coords



def align_tissue(ad_tar_coor, ad_src_coor, pca_comb_features, src_img, alpha=0.5):
    """
    Aligns the source coordinates to the target coordinates using Coherent Point Drift (CPD)
    registration, and applies a homography transformation to warp the source coordinates accordingly.

    :param ad_tar_coor: Numpy array of target coordinates to which the source will be aligned.
    :param ad_src_coor: Numpy array of source coordinates that will be aligned to the target.
    :param pca_comb_features: PCA-combined feature matrix used as additional features for the alignment process.
    :param src_img: Source image to be warped based on the alignment.
    :param alpha: Regularization parameter for CPD registration, default is 0.5.
    :return: 
        - cpd_coor: The new source coordinates after CPD alignment.
        - homo_coor: The source coordinates after applying the homography transformation.
        - aligned_image: The source image warped based on the homography transformation.
    """

    # Normalize target and source coordinates to the range [0, 1]
    ad_tar_coor_z = (ad_tar_coor - ad_tar_coor.min()) / (ad_tar_coor.max() - ad_tar_coor.min())
    ad_src_coor_z = (ad_src_coor - ad_src_coor.min()) / (ad_src_coor.max() - ad_src_coor.min())
    
    # Normalize PCA-combined features to the range [0, 1]
    pca_comb_features_z = (pca_comb_features - pca_comb_features.min()) / (pca_comb_features.max() - pca_comb_features.min())
    
    # Concatenate spatial and PCA-combined features for target and source
    target = np.concatenate((ad_tar_coor_z, pca_comb_features_z[:ad_tar_coor.shape[0], :2]), axis=1)
    source = np.concatenate((ad_src_coor_z, pca_comb_features_z[ad_tar_coor.shape[0]:, :2]), axis=1)
    
    # Initialize and run the CPD registration (deformable with regularization)
    reg = DeformableRegistration(X=target, Y=source, low_rank=True,
                                 alpha=alpha, 
                                 max_iterations=int(1e9), tolerance=1e-9)
    
    TY = reg.register()[0]  # TY contains the transformed source points

    # Rescale the CPD-aligned coordinates back to the original range of target coordinates
    cpd_coor = TY[:, :2] * (ad_tar_coor.max() - ad_tar_coor.min()) + ad_tar_coor.min()

    # Find homography transformation based on CPD-aligned coordinates and apply it
    h = find_homography_translation_rotation(ad_src_coor, cpd_coor)
    homo_coor = apply_homography(ad_src_coor, h)
    
    # Warp the source image using the computed homography
    aligned_image = cv2.warpPerspective(src_img, h, (src_img.shape[1], src_img.shape[0]))

    # Return the CPD-aligned coordinates, the homography-transformed coordinates, and the warped image
    return cpd_coor, homo_coor, aligned_image


