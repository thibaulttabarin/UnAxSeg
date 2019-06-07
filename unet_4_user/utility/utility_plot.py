#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:12:14 2019

@author: thibault
"""
import matplotlib.pyplot as plt
import numpy as np

def bland_altman_plot(m1, m2,
                      sd_limit=1.96,
                      ax=None,
                      scatter_kwds=None,
                      mean_line_kwds=None,
                      limit_lines_kwds=None):
    """
    Bland-Altman Plot.
    A Bland-Altman plot is a graphical method to analyze the differences
    between two methods of measurement. The mean of the measures is plotted
    against their difference.
    Parameters
    ----------
    m1, m2: pandas Series or array-like
    sd_limit : float, default 1.96
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted will be
                       md - sd_limit * sd, md + sd_limit * sd
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences.
        If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
        defaults to 3 standard deviatons on either side of the mean.
    ax: matplotlib.axis, optional
        matplotlib axis object to plot on.
    scatter_kwargs: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
   Returns
    -------
    ax: matplotlib Axis object
    """

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))
        
    Image.MAX_IMAGE_PIXELS = None
    im =Image.open(test_image_path)
    img=np.array(im)
    
    return img, filename
    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    if ax is None:
        ax = plt.gca()

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kwds)
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=14,
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate('-SD{}: {}'.format(sd_limit, np.round(lower, 2)),
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=14,
                    xycoords='axes fraction')
        ax.annotate('+SD{}: {}'.format(sd_limit, np.round(upper, 2)),
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    fontsize=14,
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    ax.set_ylabel('Difference', fontsize=15)
    ax.set_xlabel('Means', fontsize=15)
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    return ax

def bland_altman_plot_2(m1, m2,
                      sd_limit=1.96,
                      res_limit = 0.25,
                      ax=None,
                      scatter_kwds={'s' : 20},
                      mean_line_kwds={'color':'gray', 'linewidth':1,'linestyle':'--' },
                      limit_lines_kwds={'color':'red', 'linewidth':1,'linestyle':':' }):
    """
    Bland-Altman Plot.
    A Bland-Altman plot is a graphical method to analyze the differences
    between two methods of measurement. The mean of the measures is plotted
    against their difference.
    Parameters
    ----------
    m1, m2: pandas Series or array-like
    sd_limit : float, default 1.96
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted will be
                       md - sd_limit * sd, md + sd_limit * sd
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences.
        If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
        defaults to 3 standard deviatons on either side of the mean.
    ax: matplotlib.axis, optional
        matplotlib axis object to plot on.
    scatter_kwargs: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
   Returns
    -------
    ax: matplotlib Axis object
    """

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    if ax is None:
        ax = plt.gca()

    a1=ax.scatter(means, diffs, **scatter_kwds)
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.
    # Annotate mean line with mean difference.
    ax.annotate('mean:{}'.format(np.round(mean_diff, 2)),
                xy=(ax.get_xlim()[1]*0.99, mean_diff),
                horizontalalignment='right',
                verticalalignment='top',
                fontsize=14,
                xycoords='data')

    # +/- 0.25 line
    if not res_limit==None:
        
        ax.axhline(res_limit, color='green', linestyle='--')
        ax.axhline(-res_limit, color='green', linestyle='--')
        index_out_1 = np.logical_or(diffs >res_limit, diffs <-res_limit) 
        a2=ax.scatter(means[index_out_1], diffs[index_out_1], s = 20, color ='green') 
        percent_out_1 = 100*sum(index_out_1)/len(means)
        legend1 = '{0:.2f}%'.format(100-percent_out_1)
        
    # Limit of agreement
    limit_of_agreement = sd_limit * std_diff
    lower = mean_diff - limit_of_agreement
    upper = mean_diff + limit_of_agreement
    
    ax.axhline(lower, **limit_lines_kwds)
    ax.axhline(upper, **limit_lines_kwds)

        
    ax.annotate('-SD{}: {}'.format(sd_limit, np.round(lower, 2)),
                    xy=(ax.get_xlim()[1]*0.99, lower),
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=12, color='red',
                    xycoords='data')
    ax.annotate('+SD{}: {}'.format(sd_limit, np.round(upper, 2)),
                    xy=(ax.get_xlim()[1]*0.99, upper),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=12, color='red',
                    xycoords='data')
       
    
    index_out_2 = np.logical_or(diffs >upper, diffs <lower) 
    a3=ax.scatter(means[index_out_2], diffs[index_out_2], s = 20, color ='red')
    percent_out_2 = 100*sum(index_out_2)/len(means)
    
    legend3 = '-/+1.96sd :{0:.2f}%'.format(percent_out_2)    
    
    try:
        legend1
        legend2 = '-/+.25 :{0:.2f}%'.format(percent_out_1-percent_out_2)
        legend = [legend1, legend2, legend3]
        box = [a1,a2,a3]
    except NameError:
        legend1 = '{0:.2f}%'.format(100-percent_out_2)
        legend = [legend1, legend3]
        box = [a1,a3]
    
    plt.legend(box,legend)

    
    ax.set_ylabel('Difference', fontsize=16)
    ax.set_xlabel('Means', fontsize=16)
    #ax.tick_params(labelsize=13)
    plt.tight_layout()
    return ax, (lower,upper)

def hist_percent(x, sum_=None, bins=50, title='plot', xlabel='x', ylabel='y'):
    
    n, bins =np.histogram(x, bins)
    
    if sum_==None : n = n/sum(n)
    else : n = n/sum_
    
    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    plt.figure()
    plt.bar(bin_centers,n,bin_width, edgecolor ='k')
    plt.title (title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

from scipy import stats
def scatter_stat_binned(diam, dice, bins=50):
    
    bin_means, bin_edges, binnumber = stats.binned_statistic(diam, dice, 'mean', bins=bins)
    bin_std, bin_edges, binnumber = stats.binned_statistic(diam, dice, 'std', bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    # Statistic of dice coefficient per binned diameter
    plt.figure()
    plt.scatter(diam, dice, c='b')
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='m', lw=2,
                label='binned statistic of data')
    plt.vlines((bin_edges[:-1]+bin_edges[1:])/2, bin_means-bin_std, bin_means+bin_std, colors='m', lw=1, linestyles = 'dashed',
                label='binned statistic of data')
    ax=plt.gca()
    return ax, bin_edges
    

############################################
# Plot distribution detected not detected
    
def bar_plot_dist_1_categ(feature_, legend = 'cat_1', bins=50, display_max=False, figsize=(12,9)):
    
    n_feat, bins =np.histogram(feature_, bins=bins)
    total = sum(n_feat)
    
    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    
    fig  = plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    b1= ax.bar(bin_centers, n_feat/total, bin_width, color='blue', edgecolor='black')
    
    if display_max:
        peak = np.argmax(n_feat)
        ax.annotate('max:{0:.2f}'.format(bin_centers[peak]), xy=(1.05*bin_centers[peak],n_feat[peak]/total), color='red', size=15)
        
    legend1 = '{0}'.format(legend)
    plt.legend([b1], [legend1], fontsize=16)
    
    return bins, fig

def bar_plot_dist_2_categ(diam_, diam_1, diam_2, legend = ('cat_1', 'cat_2'), bins=50, display_max=False, **kwargs):
    
    n_diam, bins =np.histogram(diam_, bins=bins)
    n_diam_1, bins =np.histogram(diam_1, bins=bins)
    n_diam_2, _ =np.histogram(diam_2, bins=bins)
    total = sum(n_diam_1)+sum(n_diam_2)
    TP = len(diam_1)
    FN = len (diam_2)
    
    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    b1= ax.bar(bin_centers, n_diam_1/total, bin_width, color='blue', edgecolor='black')
    b2= ax.bar(bin_centers, n_diam_2/total, bin_width, bottom=n_diam_1/total,
            color='green', edgecolor='black')
    
    if display_max:    
        peak = np.argmax(n_diam)
        ax.annotate('max:{0:.2f}'.format(bin_centers[peak]), xy=(1.05*bin_centers[peak],n_diam[peak]/total), color='red')
    
    legend1 = '{0} {1:.1f}%'.format(legend[0], 100*TP/(TP+FN))
    legend2 = '{0} {1:.1f}%'.format(legend[1], 100*FN/(TP+FN))
    plt.legend([b1,b2], [legend1,legend2], fontsize=16)
    
    return bins, fig

############################################
# Plot distribution detected not detected
def bar_plot_dist_2_categ_v2(diam_, diam_1, diam_2, legend = ('cat_1', 'cat_2'), bins=50):
    
    n_diam, bins =np.histogram(diam_, bins=bins)
    n_diam_1, bins =np.histogram(diam_1, bins=bins)
    n_diam_2, _ =np.histogram(diam_2, bins=bins)
    total = sum(n_diam_1)+sum(n_diam_2)
    TP = len(diam_1)
    FN = len (diam_2)
    
    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    b1 = ax.bar(bin_centers, n_diam_1/total, bin_width, color='blue', edgecolor='black')
    b2 = ax.bar(bin_centers, n_diam_2/total, bin_width, bottom=n_diam_1/total,
            color='green', edgecolor='black')
    
    peak = np.argmax(n_diam)
    ax.annotate('max:{0:.2f}'.format(bin_centers[peak]), xy=(1.05*bin_centers[peak],n_diam[peak]/total), color='red')
    
    
    legend1 = '{0} {1:.1f}%'.format(legend[0], 100*TP/(TP+FN))
    legend2 = '{0} {1:.1f}%'.format(legend[1], 100*FN/(TP+FN))
    plt.legend([b1,b2], [legend1,legend2], fontsize=16)
    
    return bins

def bar_plot_dist_3_categ(diam_1, diam_2, diam_3, legend = ('cat_1', 'cat_2', 'cat_3'), bins=50):
    n_diam_1, bins =np.histogram(diam_1, bins=bins)
    n_diam_2, _ =np.histogram(diam_2, bins=bins)
    n_diam_3, _ =np.histogram(diam_3, bins=bins)
    total = sum(n_diam_1)+sum(n_diam_2)+sum(n_diam_3)
    TP = len(diam_1)
    FN = len (diam_2)
    
    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    plt.figure()
    b1= plt.bar(bin_centers, n_diam_1/total, bin_width, color='blue', edgecolor='black')
    b2= plt.bar(bin_centers, n_diam_2/total, bin_width, bottom=n_diam_1/total,
            color='red', edgecolor='black')
    b3= plt.bar(bin_centers, n_diam_3/total, bin_width, bottom=n_diam_2/total,
            color='green', edgecolor='black')    
    
    legend1 = '{0} {1:.1f}%'.format(legend[0], 100*TP/(TP+FN))
    legend2 = '{0} {1:.1f}%'.format(legend[1], 100*FN/(TP+FN))
    legend3 = '{0} {1:.1f}%'.format(legend[2], 100*n_diam_3/total)
    plt.legend([b1,b2,b3], [legend1,legend2, lengend3], fontsize=16)
    
    return bins

def Plot_bar_3class(df, pred, gt, lower,upper):
    '''
    plot bar plot with 3 class category based: 
        1- between lower and upper threshold for the feature of choose (pred/gt)
        2- outside thershold for the feature of choose (pred/gt)
        3- False Positve class
    input :
        df : dataframe contain all relevante metric (create by Performance_measurement.Create_df)
        pred : key of the predicted feature (feature name)
        gt : key of the corresponding ground truth feature (feature name)
        lower : lower threshold (typically 1.96sd from bland altman plot)
        upper : upper threshold (typically 1.96sd from bland altman plot)
    '''
    
    df['diff_'] =df[gt]-df[pred]
    
    diam_in_sd = df.loc[(df['diff_']>=lower) & (df['diff_']<=upper) & (df['true_positive']==1), pred]
    diam_out_sd = df.loc[((df['diff_']<lower) | (df['diff_']>upper)) & (df['true_positive']==1), pred]
    diam_FP = df.loc[df['true_positive']==0, pred]
    diam= df[pred]
    
    n_diam, bins =np.histogram(diam, bins=50)
    total = sum(n_diam)
    n_diam_in_sd, _ = np.histogram(diam_in_sd, bins=bins)
    n_diam_out_sd, _ = np.histogram(diam_out_sd, bins=bins)
    n_diam_FP, _ = np.histogram(diam_FP, bins=bins)
    
    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    plt.figure()
    b1= plt.bar(bin_centers, n_diam_in_sd/total, bin_width, color='blue', edgecolor='black')
    b2= plt.bar(bin_centers, n_diam_out_sd/total, bin_width, bottom=n_diam_in_sd/total,
            color='red', edgecolor='black')
    b3= plt.bar(bin_centers, n_diam_FP/total, bin_width, bottom=(n_diam_in_sd+n_diam_out_sd)/total,
            color='green', edgecolor='black')
    
    legend1 = 'in 1.96sd : {0:.1f}%'.format(100*sum(n_diam_in_sd/total))
    legend2 = 'out 1.96sd : {0:.1f}%'.format(100*sum(n_diam_out_sd/total))
    legend3 = 'FP : {0:.1f}%'.format(100*sum(n_diam_FP/total))
    
    plt.legend([b1,b2,b3], [legend1, legend2, legend3], fontsize=12)

############################################
# Plot distribution detected not detected
def bar_plot_dist_2_categ_v2(diam_, diam_1, diam_2, legend = ('cat_1', 'cat_2'), bins=50):
    
    n_diam, bins =np.histogram(diam_, bins=bins)
    n_diam_1, bins =np.histogram(diam_1, bins=bins)
    n_diam_2, _ =np.histogram(diam_2, bins=bins)
    total = sum(n_diam_1)+sum(n_diam_2)
    TP = len(diam_1)
    FN = len (diam_2)
    
    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    b1 = ax.bar(bin_centers, n_diam_1/total, bin_width, color='blue', edgecolor='black')
    b2 = ax.bar(bin_centers, n_diam_2/total, bin_width, bottom=n_diam_1/total,
            color='green', edgecolor='black')
    
    peak = np.argmax(n_diam)
    ax.annotate('max:{0:.2f}'.format(bin_centers[peak]), xy=(1.05*bin_centers[peak],n_diam[peak]/total), color='red')
    
    
    legend1 = '{0} {1:.1f}%'.format(legend[0], 100*TP/(TP+FN))
    legend2 = '{0} {1:.1f}%'.format(legend[1], 100*FN/(TP+FN))
    plt.legend([b1,b2], [legend1,legend2], fontsize=16)
    
    return bins

