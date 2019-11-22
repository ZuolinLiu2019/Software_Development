    import numpy as np


    def k_mean_cluster(self, image_values, k=3, initial_means=None):
        """
        Separate the provided RGB values into
        k separate clusters using the k-means algorithm,
        then return an updated version of the image
        with the original values replaced with
        the corresponding cluster values.

        params:
        image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        k = int
        initial_means = numpy.ndarray[numpy.ndarray[float]] or None

        returns:
        updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        """

        convergence = False
        image_flatten = flatten_image_matrix(image_values)
        m, n = image_flatten.shape

        if initial_means is None:
            centroids = image_flatten[np.random.randint(m, size=k), :]
        else:
            centroids = initial_means

        old_classification = np.random.choice(k,size=m)
        while not convergence:
            classification = self.expectation(image_flatten, centroids)
            centroids = self.maximization(image_flatten, classification)
            if np.sum(np.abs(classification - old_classification))==0:
                convergence = True
            old_classification = classification
        output = np.zeros((m, n))
        for x in range(k):
            output[classification==x] = centroids[x, :]

        return centroids, unflatten_image_matrix(output, image_values.shape[1])

    def __expectation__(image_values, centroids, k=k):
        m, n = image_values.shape
        distance = np.zeros((m,k))
        for i in range(k):
            distance[:,i] += np.sum((image_values-centroids[i,:])**2, axis= 1)
        classification = np.argmin(distance, axis=1)
        return classification

    def __maximization__(image_values, classification, k=k):
        _, d = image_values.shape
        centroids = np.empty((0,d))
        for i in range(k):
            image_sub = image_values[classification==i]
            centroids = np.vstack((centroids, np.mean(image_sub, axis = 0)))
        return centroids
