import numpy as np
from sklearn.model_selection import train_test_split

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

def split_dataset(TEST_RATIO, VALIDATION_RATIO, snapshots_1D, snapshots_2D):
    if TEST_RATIO == 0 and VALIDATION_RATIO == 0:
        snapshots_1D_train = snapshots_1D
        snapshots_2D_train = snapshots_2D
        snapshots_1D_test = np.array([])
        snapshots_2D_test = np.array([])
        VALIDATION_DATA = (np.array([]), np.array([]))
    elif TEST_RATIO > 0 and VALIDATION_RATIO == 0:
        snapshots_1D_train, snapshots_1D_test, snapshots_2D_train, snapshots_2D_test = train_test_split(snapshots_1D, snapshots_2D, test_size=TEST_RATIO)
        VALIDATION_DATA = (np.array([]), np.array([]))
    elif TEST_RATIO == 0 and VALIDATION_RATIO > 0:
        snapshots_1D_train, snapshots_1D_valid, snapshots_2D_train, snapshots_2D_valid = train_test_split(snapshots_1D, snapshots_2D, test_size=VALIDATION_RATIO)
        snapshots_1D_test = np.array([])
        snapshots_2D_test = np.array([])
        VALIDATION_DATA = (snapshots_1D_valid, snapshots_2D_valid)
    elif TEST_RATIO > 0 and VALIDATION_RATIO > 0:
        snapshots_1D_train, snapshots_1D_testvalid, snapshots_2D_train, snapshots_2D_testvalid = train_test_split(snapshots_1D, snapshots_2D, test_size=TEST_RATIO+VALIDATION_RATIO)
        snapshots_1D_test, snapshots_1D_valid, snapshots_2D_test, snapshots_2D_valid,= train_test_split(snapshots_1D_testvalid, snapshots_2D_testvalid, test_size=VALIDATION_RATIO/(TEST_RATIO+VALIDATION_RATIO))
        VALIDATION_DATA = (snapshots_1D_valid, snapshots_2D_valid)

    return snapshots_1D_train, snapshots_2D_train,  snapshots_1D_test, snapshots_2D_test, VALIDATION_DATA