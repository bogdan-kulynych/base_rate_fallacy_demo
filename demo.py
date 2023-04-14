import streamlit as st
import numpy as np
import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt


def plot_false_alarms(tpr, fpr, log_scale=False):
    base_rate = np.linspace(0.0, 1.0, 1000)
    ppv_vals = tpr * base_rate / (tpr * base_rate + fpr * (1.0 - base_rate))

    sns.set(style="whitegrid", context="notebook", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(base_rate * 100, (1.0 - ppv_vals) * 100)
    plt.vlines(fpr * 100, 0.0, 100, label="FPR", linestyle="--", color="lightgrey")
    plt.ylabel("Effective False-Alarm Rate, %")
    plt.legend()
    plt.xlabel("Base rate, %")
    plt.xlim(0, 50)
    plt.ylim(0, 100)
    # if log_scale:
    #     plt.xscale("symlog", base=10)
    #     ax.set_xticks([0.1, 1, 10, 100])
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    return fig


def plot_acc(tpr, fpr, log_scale=False):
    base_rate = np.linspace(0.0, 1.0, 1000)
    acc_vals = base_rate * tpr + (1 - fpr) * (1 - base_rate)

    sns.set(style="whitegrid", context="notebook", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(base_rate * 100, acc_vals * 100)
    plt.ylabel("Accuracy, %")
    plt.xlabel("Base rate, %")
    plt.xlim(0, 50)
    plt.ylim(0, 100)
    # if log_scale:
    #     plt.xscale("symlog", base=10)
    #     ax.set_xticks([0.1, 1, 10, 100])
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    return fig


if __name__ == "__main__":
    st.title("Base-Rate Fallacy and the False-Positive Paradox")
    st.write(
        """If we need to detect a rare event such as a disease or a security
        incident using a predictive model, the _false
        positive rate_ (FPR) is the critical metric that is crucial for the
        detector's quality. If the prevalence of the event – _the base rate_ – is higher
        than FPR then the overwhelming
        number of detections will be false alarms, rendering the detector useless.
        """
    )
    st.write(
        r"""
        In the following, the binary output of the detector is denoted by the random variable $D$
        ($D = +1$ means the detector predicts that the event happened and $D = -1$ means it predicts
        that it did not happen), and the event itself by $E$ ($E = +1$ means the event happens and
        $E = -1$ means it does not). The base rate is $\Pr[E = +1]$.
        """
    )

    st.sidebar.write("You can tweak the performance metrics of the detector here.")
    st.sidebar.caption(r"$\Pr[D = +1 \mid E = -1]$")
    fpr = st.sidebar.slider(
        "Detector's FPR, %",
        min_value=0.0001 * 100,
        max_value=0.5 * 100,
        value=0.1 * 100,
    )
    st.sidebar.caption(r"$\Pr[D = +1 \mid E = +1]$")
    tpr = st.sidebar.slider(
        "Detector's TPR (sensitivity)",
        min_value=0.5 * 100,
        max_value=1.0 * 100,
        value=0.99 * 100,
    )
    # log_scale = st.sidebar.checkbox("Logarithmic x axis", value=False)

    tab1, tab2 = st.tabs(["Effective False-Alarm Rate", "Accuracy"])

    tab1.write(
        r"The following plot shows the effective rate of false alarms "
        r"$\Pr[E = -1 \mid D = +1]$, "
        r"which is the inverse of the positive predictive value."
    )
    tab1.write(
        "Whenever the base rate is much lower than FPR, "
        "the majority of detections are false alarms:"
    )
    fig = plot_false_alarms(tpr / 100, fpr / 100)
    tab1.pyplot(fig)

    tab2.write(
        "The following plot shows accuracy instead. Observe how accuracy is completely misleading as "
        "it obscures the effect of drowning in false alarms."
    )
    fig = plot_acc(tpr / 100, fpr / 100)
    tab2.pyplot(fig)
