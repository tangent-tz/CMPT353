import sys

import numpy as np
import pandas as pd
import scipy.stats as stats


def print_output(t_test_result, normaltest, levene, transformed_normaltest, transformed_levene, weekly_normaltest,
                 weekly_levene, weekly_ttest, mann_whitney):
    OUTPUT_TEMPLATE = (
        "Initial T-test p-value: {initial_ttest_p:.3g}\n"
        "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
        "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
        "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
        "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
        "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
        "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
        "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
        "Mann-Whitney U-test p-value: {utest_p:.3g}"
    )

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=t_test_result.pvalue,
        initial_weekday_normality_p=normaltest[0].pvalue,
        initial_weekend_normality_p=normaltest[1].pvalue,
        initial_levene_p=levene[0].pvalue,
        transformed_weekday_normality_p=transformed_normaltest[0].pvalue,
        transformed_weekend_normality_p=transformed_normaltest[1].pvalue,
        transformed_levene_p=transformed_levene.pvalue,
        weekly_weekday_normality_p=weekly_normaltest[0].pvalue,
        weekly_weekend_normality_p=weekly_normaltest[1].pvalue,
        weekly_levene_p=weekly_levene.pvalue,
        weekly_ttest_p=weekly_ttest.pvalue,
        utest_p=mann_whitney.pvalue,
    ))


def parse_date_string_to_datetime(date_string):
    return pd.to_datetime(date_string, format='%Y-%m-%d %H:%M:%S', errors='coerce')


def remove_dates(dataframe):
    dataframe = dataframe[dataframe['date'].dt.year < 2014]
    dataframe = dataframe[dataframe['date'].dt.year > 2011]
    dataframe = dataframe[dataframe['subreddit'] == 'canada']
    return dataframe


def t_test(dataframe):
    weekend = dataframe[dataframe['is_weekend']]
    weekday = dataframe[~dataframe['is_weekend']]
    t_test_result = stats.ttest_ind(weekend['comment_count'], weekday['comment_count'])
    return t_test_result


def divide_weekend_weekday(dataframe):
    dataframe['day_of_week'] = dataframe['date'].dt.weekday
    dataframe['is_weekend'] = dataframe['day_of_week'] >= 5
    return dataframe


def normal_test(dataframe):
    weekday = dataframe[~dataframe['is_weekend']]
    weekend = dataframe[dataframe['is_weekend']]
    normaltest_weekday = stats.normaltest(weekday['comment_count'])
    normaltest_weekend = stats.normaltest(weekend['comment_count'])
    return normaltest_weekday, normaltest_weekend


def levene_test(dataframe):
    weekday = dataframe[~dataframe['is_weekend']]
    weekend = dataframe[dataframe['is_weekend']]
    levene = stats.levene(weekday['comment_count'], weekend['comment_count'])
    return levene


def histogram_plot(dataframe):
    import matplotlib.pyplot as plt
    weekday = dataframe[~dataframe['is_weekend']]
    weekend = dataframe[dataframe['is_weekend']]
    plt.hist(weekday['comment_count'], bins=30, alpha=0.5, label='weekday')
    plt.hist(weekend['comment_count'], bins=30, alpha=0.5, label='weekend')
    plt.hist(dataframe['comment_count'], bins=30, alpha=0.5, label='all')
    plt.legend(loc='upper right')
    plt.show()


def transform_data(dataframe):
    dataframe['comment_count'] = np.sqrt(dataframe['comment_count'])
    return dataframe


def transformed_normal_test(dataframe):
    weekday = dataframe[~dataframe['is_weekend']]
    weekend = dataframe[dataframe['is_weekend']]
    normaltest_weekday = stats.normaltest(weekday['comment_count'])
    normaltest_weekend = stats.normaltest(weekend['comment_count'])
    return normaltest_weekday, normaltest_weekend


def transformed_levene_test(dataframe):
    weekday = dataframe[~dataframe['is_weekend']]
    weekend = dataframe[dataframe['is_weekend']]
    levene = stats.levene(weekday['comment_count'], weekend['comment_count'])
    return levene


def grouped_weekly_data(dataframe):
    dataframe['year'] = dataframe['date'].dt.isocalendar().year
    dataframe['week'] = dataframe['date'].dt.isocalendar().week
    dataframe = dataframe.groupby(['year', 'week', 'is_weekend']).agg({'comment_count': 'mean'}).reset_index()
    return dataframe


def grouped_weekly_data_normal_test(dataframe):
    weekday = dataframe[~dataframe['is_weekend']]
    weekend = dataframe[dataframe['is_weekend']]
    normaltest_weekday = stats.normaltest(weekday['comment_count'])
    normaltest_weekend = stats.normaltest(weekend['comment_count'])
    return normaltest_weekday, normaltest_weekend


def grouped_weekly_data_levene_test(dataframe):
    weekday = dataframe[~dataframe['is_weekend']]
    weekend = dataframe[dataframe['is_weekend']]
    levene = stats.levene(weekday['comment_count'], weekend['comment_count'])
    return levene

def grouped_weekly_data_t_test(dataframe):
    weekend = dataframe[dataframe['is_weekend']]
    weekday = dataframe[~dataframe['is_weekend']]
    t_test_result = stats.ttest_ind(weekend['comment_count'], weekday['comment_count'])
    return t_test_result

def mann_whitney_u_test(dataframe):
    weekend = dataframe[dataframe['is_weekend']]
    weekday = dataframe[~dataframe['is_weekend']]
    u_test = stats.mannwhitneyu(weekend['comment_count'], weekday['comment_count'])
    return u_test
def main(file_param):
    counts = pd.read_json(file_param, lines=True)
    counts['date'] = parse_date_string_to_datetime(counts['date'])
    counts = remove_dates(counts)
    counts = divide_weekend_weekday(counts)
    t_test_result = t_test(counts)
    normaltest = normal_test(counts)
    levene = levene_test(counts),
    histogram_plot(counts)
    transformed = transform_data(counts)
    transformed_normaltest = transformed_normal_test(transformed)
    transformed_levene = transformed_levene_test(transformed)
    fresh_counts = pd.read_json(file_param, lines=True)
    fresh_counts['date'] = parse_date_string_to_datetime(fresh_counts['date'])
    fresh_counts = remove_dates(fresh_counts)
    fresh_counts = divide_weekend_weekday(fresh_counts)
    weekly = grouped_weekly_data(fresh_counts)
    weekly_normaltest = grouped_weekly_data_normal_test(weekly)
    weekly_levene = grouped_weekly_data_levene_test(weekly)
    weekly_ttest = grouped_weekly_data_t_test(weekly)
    mann_whitney = mann_whitney_u_test(counts)
    print_output(t_test_result, normaltest, levene, transformed_normaltest, transformed_levene, weekly_normaltest,
                 weekly_levene, weekly_ttest, mann_whitney)


if __name__ == "__main__":
    file_param = sys.argv[1]

    main(file_param)
