import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_style("whitegrid")


class EDAPlots:

    @staticmethod
    def correlation_heatmap(df):

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            df.corr(
                numeric_only=True
            ),
            cmap="coolwarm",
            ax=ax
        )

        plt.title(
            "Correlation Heatmap"
        )

        return fig

    @staticmethod
    def feature_histogram(
        df,
        feature
    ):

        fig = px.histogram(
            df,
            x=feature,
            nbins=50,
            title=f"{feature} Distribution"
        )

        return fig

    @staticmethod
    def feature_boxplot(
        df,
        feature
    ):

        fig = px.box(
            df,
            y=feature,
            title=f"{feature} Boxplot"
        )

        return fig

    @staticmethod
    def feature_scatter(
        df,
        x_feature,
        y_feature
    ):

        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            title=f"{x_feature} vs {y_feature}"
        )

        return fig

    @staticmethod
    def feature_lineplot(
        df,
        feature
    ):

        fig = px.line(
            df,
            y=feature,
            title=f"{feature} Trend"
        )

        return fig

    @staticmethod
    def feature_violin(
        df,
        feature
    ):

        fig = px.violin(
            df,
            y=feature,
            box=True,
            points="outliers",
            title=f"{feature} Violin Plot"
        )

        return fig

    @staticmethod
    def pair_plot(df):

        numeric_cols = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        numeric_cols = numeric_cols[:4]

        if len(numeric_cols) < 2:
            return None

        sample_df = df.sample(
            min(
                500,
                len(df)
            ),
            random_state=42
        )

        fig = px.scatter_matrix(
            sample_df,
            dimensions=numeric_cols
        )

        return fig

    @staticmethod
    def correlation_bar(df):

        corr_matrix = (
            df.corr(
                numeric_only=True
            )
            .abs()
        )

        corr_values = (
            corr_matrix.unstack()
            .sort_values(
                ascending=False
            )
        )

        corr_values = corr_values[
            corr_values < 1
        ]

        top_corr = corr_values.head(20)

        fig = px.bar(
            x=[
                f"{i[0]} - {i[1]}"
                for i in top_corr.index
            ],
            y=top_corr.values,
            title="Top Correlations"
        )

        return fig

    @staticmethod
    def missing_values_plot(df):

        missing = df.isnull().sum()

        missing = missing[
            missing > 0
        ]

        if len(missing) == 0:
            return None

        fig = px.bar(
            x=missing.index,
            y=missing.values,
            title="Missing Values"
        )

        return fig

    @staticmethod
    def anomaly_distribution(df):

        if "Anomaly" not in df.columns:
            return None

        counts = (
            df["Anomaly"]
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            "Anomaly",
            "Count"
        ]

        fig = px.pie(
            counts,
            names="Anomaly",
            values="Count",
            title="Anomaly Distribution"
        )

        return fig