import numpy


filename = 'abalone.data.txt'

with open(filename) as f:
    reader = csv.reader(f)
    data = list(reader)

sex = [point[1] for point in data]
weight = [point[5] for point in data]
points = [(float(p[2]), float(p[5])) for p in data]
# import matplotlib.pyplot


filename = 'zoo.data.txt'

with open(filename) as f:
    reader = csv.reader(f)
    data = list(reader)

legs = [p[13] for p in points]
predator = [p[7] for p in points]
points = [(float(p[13]), float(p[7])) for p in data]

legss = df['legs']
predators = df['animalype']

# numpy.random.seed(5)
# num_observations = 2000
# #
# x1 = np.random.multivariate_normal(, num_observations)
# x2 = np.random.multivariate_normal(, num_observations)




# link function credit to https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(points):
    return 1 / (1 + numpy.exp(-points))

#maximize the likelihood
#training data set
def log_likelihood(x, y, z):
    s = numpy.dot(x, z)
    return numpy.sum( y*s - numpy.log(1 + numpy.exp(s)) )

#the gradient of likelihood function
def gradient_ (target, predictions, features):
    return numpy.dot(features.T, target - predictions)


def logistic_regression(features, target, steps, learning_rate, intercepts = False):
    if intercepts:
        #fill the matrix with 1s
        intercept = numpy.ones((features.shape[0], 1))
        #xip the intercept with features
        features = numpy.hstack((intercept, features))

    weights = numpy.zeros(features.shape[1])

    for x in xrange(steps):
        scores = numpy.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient, update steps
        gradient = gradient_(target, predictions, feature)
        weights += learning_rate * gradient

        #PRINT DATA EVERY 5000 STEPS
        #ONLY THING THAT CHANGES OVER TIME IS WEIGHTS, IT KEEPS UPDATING
        if x % 5000 == 0:
            print log_likelihood(features, target, weights)

    return weights

weights = logistic_regression(weight, sex, 300000, 5e-4, True)
