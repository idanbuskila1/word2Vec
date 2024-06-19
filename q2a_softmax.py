import numpy as np
import scipy
import scipy.special


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        x -= np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)

    else:
        # Vector
        x -= np.max(x)
        x = np.exp(x)
        x /= np.sum(x)

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([[0.26894142, 0.73105858], [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    num_rand_tests = int(1e5)
    for _ in range(num_rand_tests):
        shape = np.random.randint(1, 20, size=2)
        x = np.random.random(shape) * np.random.randint(1, 1000)
        real_softmax_x = scipy.special.softmax(x, axis=1)
        my_softmax_x = softmax(x)
        assert np.allclose(real_softmax_x, my_softmax_x, rtol=1e-05, atol=1e-06)
    print(f"All {num_rand_tests} passed!")


if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
