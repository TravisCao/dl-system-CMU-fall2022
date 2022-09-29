#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t sample_i = 0; sample_i < m; sample_i += batch)
    {
        size_t size = 0;
        size_t n_row, n_col = 0;
        if (sample_i + batch > m)
        {
            size = m - sample_i;
        }
        else
        {
            size = batch;
        }
        float x_batch[size * n];
        unsigned char y_batch[size];

        n_row = size;
        n_col = n;
        for (size_t i = 0; i < n_row; i++)
        {
            for (size_t j = 0; j < n_col; j++)
            {
                x_batch[i * n_col + j] = X[(sample_i + i) * n_col + j];
            }
            y_batch[i] = y[sample_i + i];
        }

        float x_theta_mul[size * k] = {0};
        float Z[size * k] = {0};
        float I_y[size * k] = {0};

        // x_theta_mul (size * n) * (n * k) -> (size * k)
        n_row = size;
        n_col = k;
        for (size_t i = 0; i < n_row; i++)
        {
            for (size_t j = 0; j < n_col; j++)
            {
                for (size_t jj = 0; jj < n; jj++)
                {
                    x_theta_mul[i * n_col + j] += x_batch[i * n + jj] * theta[jj * k + j];
                }
            }
        }

        // Z is of size (size * k)
        n_row = size;
        n_col = k;
        for (size_t i = 0; i < n_row; i++)
        {
            float row_sum = 1e-8;
            for (size_t j = 0; j < n_col; j++)
            {
                Z[i * n_col + j] = std::exp(x_theta_mul[i * n_col + j]);
                row_sum += Z[i * n_col + j];
            }
            for (size_t j = 0; j < n_col; j++)
            {
                Z[i * n_col + j] /= row_sum;
            }
        }

        for (size_t i = 0; i < size; i++)
        {
            int label = int(y_batch[i]);
            I_y[i * k + label] = 1;
        }

        float x_batch_T[n * size] = {0};
        float x_batch_T_mul[n * k] = {0};

        n_row = size;
        n_col = n;
        for (size_t i = 0; i < n_row; i++)
        {
            for (size_t j = 0; j < n_col; j++)
            {
                x_batch_T[j * n_row + i] = x_batch[i * n_col + j];
            }
        }

        // x_batch_T_mul: x_batch.T * (Z - I_y)
        //               (n * size) * (size * k) -> (n * k)
        n_row = n;
        n_col = k;
        for (size_t i = 0; i < n_row; i++)
        {
            for (size_t j = 0; j < n_col; j++)
            {
                for (size_t jj = 0; jj < size; jj++)
                {
                    x_batch_T_mul[i * n_col + j] += x_batch_T[i * size + jj] * (Z[jj * k + j] - I_y[jj * k + j]);
                }
                theta[i * n_col + j] -= lr / int(size) * x_batch_T_mul[i * n_col + j];
            }
        }
    }
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
