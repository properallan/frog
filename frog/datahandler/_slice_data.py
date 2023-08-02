import numpy as np

def sliceDataAlongAxis(data, fractions, axis):
    data_size = data.shape[0]
    fractions_ = np.zeros_like(fractions, dtype=int)

    total_size = 0
    for i, fraction in enumerate(fractions):
        total_size += int(data_size*fraction)
    remain = data_size-total_size

    slices = ()
    for i, fraction in enumerate(fractions):
        fractions_[i] = int(data_size*fraction)
        if i > 0:
            fractions_[i] += fractions_[i-1]
            slice = data.take(range(fractions_[i-1], fractions_[i]), axis)

        else:
            slice = data.take(range(0, fractions_[i]+remain), axis)

        slices += (slice,)

    return slices