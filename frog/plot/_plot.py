import matplotlib.pyplot as plt

def plot_results(hfdh, fr, lf_test, hf_test, idx, ax=None, ylim=None, subplots_kwarg={}):
    if ax is None:
        fig, ax = plt.subplots()
    elif subplots_kwarg != {}:
        fig, ax = plt.subplots(**subplots_kwarg)
        ax = ax.flatten()

        for i in range(len(ax)):
            plot_results(hfdh, fr, lf_test, hf_test, idx[i], ax=ax[i], ylim=ylim)
    else:
        ax.plot(hfdh(hf_test)[idx,'Heat_Flux'][1:-1])
        ax.plot(hfdh(fr.predict(lf_test))[idx,'Heat_Flux'][1:-1], ls='-.')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()