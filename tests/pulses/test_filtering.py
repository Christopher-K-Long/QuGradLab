import numpy as np

from qugradlab.pulses.filtering import get_fixed_filter, get_time_response, get_frequency_response, const_spline_time_spline_matrix, apply_filtering_transform

# This will test the results are the same for first-order high- and low-pass
#   filters

TIME_CONSTANT = 1.34
POWERS = np.array([0, 1, 2, 3])

TRANSFER_FUNCTIONS = [
    lambda w: 1 / (1 + 1j * w * TIME_CONSTANT * 2 * np.pi), # low-pass
    lambda w: 1j * w * TIME_CONSTANT * 2 * np.pi / (1 + 1j * w * TIME_CONSTANT * 2 * np.pi), # high-pass
]

def test_get_time_response():
    sample_times = np.linspace(-100, 100, 1000000)
    for power in POWERS:
        signal = np.power(sample_times, power) * (sample_times >= 0)
        ft_signal = np.fft.fft(signal)
        w = np.fft.fftfreq(len(signal), d=sample_times[1] - sample_times[0])
        for m, transfer_function in enumerate(TRANSFER_FUNCTIONS):
            new_ft_signal = ft_signal * transfer_function(w)
            new_signal = np.fft.ifft(new_ft_signal).real
            response = get_time_response(sample_times,
                                         TIME_CONSTANT,
                                         TIME_CONSTANT,
                                         1,
                                         power,
                                         m)
            range_to_check = np.logical_and(0.1 <= np.abs(sample_times), np.abs(sample_times) <= 50), 
            assert np.allclose(new_signal[range_to_check], response[range_to_check], atol=1e-4)

def test_frequency_response():
    np.random.seed(0)
    time_resposne = np.random.rand(3, 2, 100)
    result = get_frequency_response(time_resposne)
    time_resposne = np.concatenate([time_resposne,
                                    np.zeros_like(time_resposne)],
                                   axis=-1)
    expected = np.fft.fft(time_resposne, axis=-1)
    assert np.allclose(expected, result)

def test_const_spline_time_spline_matrix():
    np.random.seed(0)
    points = np.random.rand(10, 3)
    time_spline_matrix = const_spline_time_spline_matrix(points)
    expected = np.expand_dims(np.concatenate([points[:1],
                                              np.diff(points, axis=0)],
                                             axis=0), axis=1)
    assert np.array_equal(time_spline_matrix,
                          expected)

def test_apply_filtering_transform():
    np.random.seed(0)
    n_pieces_in_spline = 3
    n_orders = 1
    samples_per_piece = 2
    points = np.random.rand(n_pieces_in_spline, 4)
    time_spline_matrix = const_spline_time_spline_matrix(points)
    time_response = np.random.rand(n_orders, samples_per_piece, n_pieces_in_spline)
    frequency_response = get_frequency_response(time_response)
    result = apply_filtering_transform(time_spline_matrix, frequency_response)
    time_response = time_response.T
    expected = []
    for i in range(points.shape[-1]):
        samples =  []
        for j in range(n_orders):
            convolutions = []
            for k in range(samples_per_piece):
                convolutions.append(
                    np.sum([time_spline_matrix[l, j, i]
                           *np.concatenate([np.zeros(l),
                                            time_response[:n_pieces_in_spline-l,
                                                          k,
                                                          j]])
                            for l in range(n_pieces_in_spline)],
                            axis=0))
            samples.append(convolutions)
        expected.append(np.sum(np.array(samples), axis=0).T.flatten())
    expected = np.array(expected).T
    assert np.allclose(result, expected)

def test_get_fixed_filter():
    n_pieces_in_spline = 10
    samples_per_piece = 10000
    T = 100
    points = np.random.rand(n_pieces_in_spline, 4)
    sample_times = np.linspace(0, T, n_pieces_in_spline * samples_per_piece)
    signal = np.concatenate([points, points[-1:]])[np.floor(n_pieces_in_spline * sample_times / T).astype(int)]
    ft_signal = np.fft.fft(signal, axis=0)
    w = np.fft.fftfreq(len(signal), d=sample_times[1] - sample_times[0])
    for m, transfer_function in enumerate(TRANSFER_FUNCTIONS):
        new_ft_signal = ft_signal * np.expand_dims(transfer_function(w), -1)
        new_signal = np.fft.ifft(new_ft_signal, axis=0).real
        filter_function = get_fixed_filter(n_pieces_in_spline,
                                           T/n_pieces_in_spline,
                                           samples_per_piece,
                                           TIME_CONSTANT,
                                           TIME_CONSTANT,
                                           1,
                                           np.array([0]),
                                           m)
        result = filter_function(const_spline_time_spline_matrix(points))
        range_to_check = 20 <= sample_times
        assert np.allclose(new_signal[range_to_check], result[range_to_check], atol=1e-3)