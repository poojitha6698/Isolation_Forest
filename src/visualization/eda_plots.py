import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_style("whitegrid")


class EDAPlots:

    @staticmethod
    def fraud_distribution(df):

        fig = px.pie(
            df,
            names="Class",
            title="Fraud vs Non-Fraud Distribution"
        )

        return fig

    @staticmethod
    def amount_distribution(df):

        fig = px.histogram(
            df,
            x="Amount",
            nbins=100,
            title="Transaction Amount Distribution"
        )

        return fig

    @staticmethod
    def amount_by_class(df):

        fig = px.box(
            df,
            x="Class",
            y="Amount",
            title="Amount Distribution by Class"
        )

        return fig

    @staticmethod
    def time_distribution(df):

        fig = px.histogram(
            df,
            x="Time",
            nbins=100,
            title="Transaction Time Distribution"
        )

        return fig

    @staticmethod
    def correlation_heatmap(df):

        fig, ax = plt.subplots(
            figsize=(15, 10)
        )

        sns.heatmap(
            df.corr(),
            cmap="coolwarm",
            ax=ax
        )

        plt.title(
            "Correlation Heatmap"
        )

        return fig

    @staticmethod
    def top_correlated_features(df):

        corr = df.corr()["Class"]

        corr = corr.drop(
            "Class"
        )

        corr = corr.abs()

        corr = corr.sort_values(
            ascending=False
        )

        top_corr = corr.head(10)

        fig = px.bar(
            x=top_corr.index,
            y=top_corr.values,
            title="Top Features Correlated with Fraud"
        )

        return fig

    @staticmethod
    def feature_histogram(df, feature):

        fig = px.histogram(
            df,
            x=feature,
            color="Class",
            barmode="overlay",
            title=f"{feature} Distribution"
        )

        return fig

    @staticmethod
    def feature_boxplot(df, feature):

        fig = px.box(
            df,
            x="Class",
            y=feature,
            title=f"{feature} vs Fraud"
        )

        return fig

    @staticmethod
    def pair_plot(df):

        sample_df = df.sample(
            1000,
            random_state=42
        )

        fig = px.scatter_matrix(
            sample_df,
            dimensions=[
                "V1",
                "V2",
                "V3",
                "V4"
            ],
            color="Class"
        )

        return fig

    @staticmethod
    def fraud_amount_scatter(df):

        sample_df = df.sample(
            5000,
            random_state=42
        )

        fig = px.scatter(
            sample_df,
            x="Time",
            y="Amount",
            color="Class",
            title="Time vs Amount"
        )

        return fig