class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)
        self.img_height, self.img_width = self.image_matrix.shape[0], self.image_matrix.shape[1]
        self.flatten_image = flatten_image_matrix(self.image_matrix
                            ).reshape((self.img_height*self.img_width, 1))
        self.num_data = self.img_height * self.img_width

    def probability_log(self, val):
        """ helper function to calculate the log probability for each component
        for all data points.
        """
        if not isinstance(val, np.ndarray):
            return np.log(self.mixing_coefficients)
                            - (0.5*np.log(2.0*math.pi*self.variances)
                            + ((val-self.means)**2/(2.0*self.variances)))
        prob_log = np.empty((val.shape[0], 0))
        for i in range(self.num_components):
            prob_log = np.hstack((prob_log, np.log(self.mixing_coefficients[i])
                                    - (0.5*np.log(2.0*math.pi*self.variances[i])
                                    + ((val-self.means[i])**2/(2.0*self.variances[i])))))
        return prob_log


    def joint_prob(self, val):
        """Calculate the joint log probability of a greyscale value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        if not isinstance(val, np.ndarray):
            return logsumexp(self.probability_log(val))
        return logsumexp(self.probability_log(val), axis=1).reshape(self.num_data, 1)

    def initialize_training(self):
        """
        Initialize the training process by setting each component mean to a random
        pixel's value (without replacement), each component variance to 1, and
        each component mixing coefficient to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before train_model() in order for tests
        to execute correctly.
        """
        self.variances = np.ones(self.num_components)
        self.mixing_coefficients = np.ones(self.num_components)/self.num_components
        index_means = np.random.choice(self.num_data, size=self.num_components, replace=False)
        self.means = self.flatten_image[index_means]

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
        converge = False
        prev_likelihood = self.likelihood()
        conv_ctr = 0
        while not converge:
            joint_prob = self.joint_prob(self.flatten_image)
            prob_log = self.probability_log(self.flatten_image)
            gamma = np.exp(prob_log - joint_prob)

            for i in range(self.num_components):
                Nk = np.sum(gamma[:,i])
                prob_i = gamma[:,i].reshape((self.num_data, 1))
                self.means[i] = np.sum(prob_i*self.flatten_image)/Nk
                self.variances[i] = np.sum(prob_i*(self.flatten_image-self.means[i])**2)/Nk
                self.mixing_coefficients[i] = Nk/(self.num_data)

            new_likelihood = self.likelihood()
            conv_ctr, converge = convergence_function(prev_likelihood, new_likelihood, conv_ctr,
                                            conv_ctr_cap=10)
            prev_likelihood = new_likelihood
        return self.means, self.variances, self.mixing_coefficients, prev_likelihood


    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """

        prob_log = self.probability_log(self.flatten_image)
        component_map = np.argmax(prob_log, axis=1)
        means = np.ones((self.num_data, 1)) * self.means.reshape((1, self.num_components))
        segment = means[component_map]
        return unflatten_image_matrix(segment, self.img_width)


    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        return np.sum(self.joint_prob(self.flatten_image))


    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        means, variances, mixing_coefficients, likelihoods = [0.0]*iters, [1.]*iters, [1.]*iters, [0.0]*iters
        for i in range(iters):
            means[i], variances[i], mixing_coefficients[i], likelihoods[i] = self.train_model()
        index_max = np.argmax(np.array(likelihoods))
        self.means = means[index_max]
        self.variances = variances[index_max]
        self.mixing_coefficients = mixing_coefficients[index_max]
        return self.segment()
