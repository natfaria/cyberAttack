import pandas as pd
from typing import Optional, Tuple, Union, List


#from cyberattacks.eda_utils import get_feature_importances


#################################################
#      BIVARIATE DISTRIBUTION OF ATTACKS        #
#################################################
"""
Plots the bivariate distribution between a selected feature and security events,
allowing visualization of patterns and trends in cyberattacks.

This chart combines **bars for the volume of incidents** and **a line for the attack rate**,
making it easier to identify risk patterns and behavior trends.

Parameters
----------
df : pd.DataFrame
    DataFrame containing the cybersecurity data to be analyzed.
feature : str
    Name of the independent variable (e.g., protocol, attack type, source IP).
target_variable : str
    Name of the target variable indicating whether an attack occurred.
attack_rate : float
    Overall attack rate of interest (e.g., average attack detection rate).
convert_str : bool, optional (default=True)
    If True, converts the feature variable to a string.
sort_values : bool, optional (default=False)
    If True, sorts values in descending order by incident volume.
figsize : Tuple[int, int], optional (default=(16,4))
    Figure size of the plot.
bar_width : float, optional (default=0.3)
    Width of the bars in the chart.
y_labelsize : int, optional (default=10)
    Font size for the Y-axis label.
yticks_labelsize : int, optional (default=10)
    Font size for the Y-axis tick labels.
show_line_labels : bool, optional (default=True)
    If True, displays labels on the attack rate line.
show_bar_labels : bool, optional (default=False)
    If True, displays labels on top of the bars.
label_fontsize : int, optional (default=9)
    Font size for labels.
rotation : int, optional (default=45)
    Rotation angle for X-axis labels.
ha : str, optional (default='right')
    Alignment of X-axis labels.
is_category_axes : bool, optional (default=False)
    If True, treats the X-axis as categorical.
custom_xticks : list or None, optional (default=None)
    Allows defining custom labels for the X-axis.
ncol : int, optional (default=4)
    Number of columns in the legend.
bbox_to_anchor : Tuple[float, float, float, float], optional (default=(0, 1.08, 1., .1))
    Position of the legend in the chart.
leg_fontsize : int, optional (default=10)
    Font size of the legend.
title_fontsize : int, optional (default=14)
    Font size of the title.
title_pad : int, optional (default=40)
    Spacing between the title and the plot.
xticks_labelsize : int, optional (default=10)
    Font size of the X-axis tick labels.
title : str, optional (default=None)
    Chart title. If None, uses a default title.
label1 : str, optional (default="Detected Attacks")
    Label for attack occurrences.
label2 : str, optional (default="Non-Attack Events")
    Label for non-attack occurrences.

Returns
-------
None
    The function displays a bivariate plot without returning values.

Example Usage
--------------
plot_attack_distribution(
    df=data, 
    feature='Protocol', 
    target_variable='Attack Detected', 
    attack_rate=0.15
)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List

def plot_attack_distribution(
    df: pd.DataFrame,
    feature: str,
    target_variable: str,
    attack_rate: float,
    convert_str: Optional[bool] = True,
    sort_values: Optional[bool] = False,
    figsize: Optional[Tuple[int, int]] = (16, 4),
    bar_width: Optional[float] = 0.3,
    y_labelsize: Optional[int] = 10,
    yticks_labelsize: Optional[int] = 10,
    show_line_labels: Optional[bool] = True,
    show_bar_labels: Optional[bool] = False,
    label_fontsize: Optional[int] = 9,
    rotation: Optional[int] = 45,
    ha: Optional[str] = 'right',
    is_category_axes: Optional[bool] = False,
    custom_xticks: Optional[Union[None, List[str]]] = None,
    ncol: Optional[int] = 4,
    bbox_to_anchor: Optional[Tuple[float, float, float, float]] = (0, 1.08, 1., .1),
    leg_fontsize: Optional[int] = 10,
    title_fontsize: Optional[int] = 14,
    title_pad: Optional[int] = 40,
    xticks_labelsize: Optional[int] = 10,
    title: str = None,
    label1: str = "Detected Attacks",
    label2: str = "Non-Attack Events"
) -> None:
    """
    Plots the relationship between a selected cybersecurity feature and attack occurrences.
    The chart combines bar and line plots for better visualization of trends.
    """

    if not title:
        title = f'Attack Rate Variation by Feature: {feature}'
    
    df_tmp = df[[feature, target_variable]].copy()
    df_tmp['count'] = 1

    if convert_str:
        df_tmp[feature] = df_tmp[feature].astype(str)

    df_tmp = df_tmp.groupby(feature)[['count', target_variable]].sum().reset_index()

    if sort_values:
        df_tmp.sort_values(by=['count'], ascending=False, inplace=True)
    else:
        df_tmp.sort_values(by=[feature], inplace=True)

    df_tmp['non_attack_count'] = (df_tmp['count'] - df_tmp[target_variable]).astype(int)
    df_tmp['attack_rate'] = df_tmp[target_variable] / df_tmp['count']
    df_tmp['percentage_traffic'] = 100 * df_tmp['count'] / df_tmp['count'].sum()
    df_tmp[target_variable] = df_tmp[target_variable].astype(int)

    fig, ax1 = plt.subplots(figsize=figsize)

    #################################
    #         PLOTTING EVENTS       #
    #################################

    x_loc = np.arange(len(df_tmp))

    ax1.bar(x_loc - bar_width / 2, df_tmp[target_variable], width=bar_width, color="darkred",
        linestyle='-', linewidth=4., alpha=0.6, label=label1)
    
    ax1.bar(x_loc + bar_width / 2, df_tmp['non_attack_count'], width=bar_width, color="gray",
        linestyle='-', linewidth=4., alpha=0.6, label=label2)

    plt.ylabel('Number of Events', fontsize=y_labelsize)

    #################################
    #        PLOTTING ATTACK RATE   #
    #################################
    ax2 = ax1.twinx()
    y_values = round(100.00 * df_tmp['attack_rate'], 2)

    ax2.plot(df_tmp[feature], y_values, linestyle='-', marker='.', linewidth=1, color="blue")

    plt.ylabel("Attack Rate [%]", fontsize=y_labelsize)
    ax2.set_ylim([0, round(y_values.max() * 1.1, 2)])

    if show_line_labels:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        for x, y in enumerate(y_values):
            ax2.text(x, y + 0.1, f'{round(y, 2)}%', ha='center', va='bottom', color="blue", weight='bold', bbox=props)

    #################################
    #       BASELINE ATTACK RATE    #
    #################################
    base_rate = round(100 * attack_rate, 2)
    plt.axhline(base_rate, color="black", linestyle='--', alpha=0.7)

    plt.legend(
        loc='upper center',
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0.5,
        frameon=False,
        fontsize=leg_fontsize
    )

    plt.title(title, fontsize=title_fontsize, weight='medium', pad=title_pad)
    plt.show() 