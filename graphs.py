"""Regenerate all notebook visualizations for the UI 'View All Graphs' modal.

Produces the same charts as heart_disease_prediction.ipynb (restyled in purple)
plus dedicated charts for the new weight / smoking / diabetes factors.
"""

import base64
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "train.csv"
TEST_CSV = ROOT / "data" / "test.csv"
MODEL_PATH = ROOT / "heart_model.pkl"

CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
PURPLES = ["#D59EDE", "#C87DDE", "#B85FDB", "#A83FD8", "#9820D4"]
PURPLE = "#9820D4"
LIGHT_PURPLE = "#E9D5FF"

sns.set_style("whitegrid")


def _load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    return train, test


def _encode(df):
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["age"], bins=[0, 40, 55, 100], labels=[0, 1, 2]).astype(int)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df


def _titles_and_target(train):
    gender_labels = ["Female", "Male"]
    cp_labels = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"]
    status_labels = ["No Disease", "Disease"]
    return gender_labels, cp_labels, status_labels


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _chart(title, fig):
    return {"title": title, "image": _fig_to_base64(fig)}


# ---------------------------------------------------------------- charts ----

def chart_numeric_distributions(train):
    df = train[["age", "trestbps", "chol", "thalach", "oldpeak"]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#1a0b2e")
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], color=PURPLES[i], edgecolor="white", bins=20)
        axes[i].set_title(col, color="white", fontsize=14)
        axes[i].tick_params(colors="white")
        for spine in axes[i].spines.values():
            spine.set_color("#4c2a6b")
    fig.delaxes(axes[-1])
    fig.tight_layout()
    return _chart("Numeric Feature Distributions", fig)


def chart_gender_cp(train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a0b2e")
    sns.countplot(data=train, x="target", hue="sex", palette="BuPu", ax=axes[0])
    axes[0].set_title("Heart Disease Count by Gender", color="white")
    axes[0].set_xlabel("Target (0 = No Disease, 1 = Disease)", color="white")
    axes[0].set_ylabel("Count", color="white")
    axes[0].legend(title="Sex", labels=["Female", "Male"])
    sns.barplot(data=train, x="cp", y="target", hue="sex", palette="RdPu", ax=axes[1])
    axes[1].set_title("Disease Rate by Chest Pain Type and Sex", color="white")
    axes[1].set_xlabel("Chest Pain Type (cp)", color="white")
    axes[1].set_ylabel("Disease Rate", color="white")
    for ax in axes:
        ax.legend(title="Sex", labels=["Female", "Male"], facecolor="#2a1647", edgecolor="#4c2a6b")
        _style_axis(ax)
    fig.tight_layout()
    return _chart("Heart Disease Count by Gender & Chest Pain Type", fig)


def chart_age_chol(train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a0b2e")
    sns.histplot(data=train, x="age", hue="target", kde=True, multiple="stack", palette="RdPu", ax=axes[0])
    axes[0].set_title("Age Distribution by Heart Disease Status", color="white")
    sns.histplot(data=train, x="chol", hue="target", kde=True, multiple="stack", palette="Purples", ax=axes[1])
    axes[1].set_title("Cholesterol Distribution by Heart Disease Status", color="white")
    for ax in axes:
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#4c2a6b")
        if ax.legend_:
            ax.legend(facecolor="#2a1647", edgecolor="#4c2a6b")
            for text in ax.legend_.get_texts():
                text.set_color("white")
    fig.tight_layout()
    return _chart("Age & Cholesterol Distribution", fig)


def chart_target_count(train):
    target_count = train["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#1a0b2e")
    target_count.plot(kind="bar", color=["#C87DDE", "#9820D4"], ax=ax, width=0.7)
    ax.set_title("Heart Disease Count", color="white", fontsize=15)
    ax.set_xticklabels(["No Disease", "Disease"], rotation=0, color="white")
    ax.set_xlabel("Status", color="white")
    ax.set_ylabel("Counts", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", color="white", fontsize=12,
        )
    fig.tight_layout()
    return _chart("Heart Disease Count", fig)


def chart_target_pie(train):
    target_count = train["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#1a0b2e")
    ax.pie(
        target_count,
        labels=["No Disease", "Disease"],
        autopct="%1.1f%%",
        colors=["#C87DDE", "#9820D4"],
        startangle=90,
        textprops={"color": "white", "fontsize": 12},
        wedgeprops={"edgecolor": "#1a0b2e", "linewidth": 2},
    )
    ax.set_title("Proportion of Patients With vs Without Heart Disease", color="white")
    fig.tight_layout()
    return _chart("Heart Disease Proportion", fig)


def chart_gender_bar(train):
    gender_target = train.groupby(["sex", "target"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#1a0b2e")
    gender_target.plot(kind="bar", color=["#C87DDE", "#9820D4"], width=0.7, ax=ax)
    ax.set_title("Heart Disease Count By Gender", color="white", fontsize=15)
    ax.set_xticklabels(["Female", "Male"], rotation=0, color="white")
    ax.set_xlabel("Gender", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    ax.legend(labels=["No Disease", "Disease"], facecolor="#2a1647", edgecolor="#4c2a6b")
    for text in ax.legend_.get_texts():
        text.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("Heart Disease Count by Gender", fig)


def chart_cp_rate(train):
    cp_target = train.groupby("cp")["target"].mean()
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a0b2e")
    cp_target.plot(kind="bar", color=PURPLES, width=0.7, ax=ax)
    ax.set_title("Heart Disease Rate by Chest Pain Type", color="white", fontsize=15)
    ax.set_xticklabels(
        ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"],
        rotation=15, color="white",
    )
    ax.set_xlabel("Chest Pain Type", color="white")
    ax.set_ylabel("Disease Rate", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("Heart Disease Rate by Chest Pain Type", fig)


def chart_age_hist(train):
    bins = 20
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a0b2e")
    train[train["target"] == 1]["age"].plot(
        kind="hist", bins=bins, color="#C87DDE", alpha=0.6, label="Disease", edgecolor="white", ax=ax
    )
    train[train["target"] == 0]["age"].plot(
        kind="hist", bins=bins, color="#9820D4", alpha=0.6, label="No Disease", edgecolor="white", ax=ax
    )
    ax.set_title("Age Distribution by Heart Disease Status", color="white", fontsize=15)
    ax.set_xlabel("Age", color="white")
    ax.set_ylabel("Frequency", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2a1647", edgecolor="#4c2a6b")
    for text in ax.legend_.get_texts():
        text.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("Age Distribution by Heart Disease Status", fig)


def chart_corr_heatmap(train):
    corr = train.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#1a0b2e")
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdPu", linewidths=0.5,
        cbar_kws={"label": "Correlation"}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", color="white", fontsize=15)
    ax.tick_params(colors="white", labelsize=8)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Correlation", color="white")
    cbar.outline.set_edgecolor("#4c2a6b")
    fig.tight_layout()
    return _chart("Feature Correlation Heatmap", fig)


def chart_boxplots(train):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#1a0b2e")
    for i, col in enumerate(["age", "trestbps", "chol", "thalach"]):
        sns.boxplot(
            data=train, x="target", y=col, hue="target",
            palette=PURPLES[:2], ax=axes[i], legend=False,
        )
        axes[i].set_title(col, color="white", fontsize=13)
        axes[i].set_xticklabels(["No Disease", "Disease"], rotation=0, color="white")
        axes[i].tick_params(colors="white")
        for spine in axes[i].spines.values():
            spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("Boxplots of Key Numeric Features by Target", fig)


def chart_thalach_violin(train):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1a0b2e")
    sns.violinplot(data=train, x="target", y="thalach", hue="target", palette="RdPu", legend=False, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"], color="white")
    ax.set_title("Max Heart Rate Achieved by Heart Disease Status", color="white", fontsize=15)
    ax.set_xlabel("Target", color="white")
    ax.set_ylabel("Max Heart Rate (thalach)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("Max Heart Rate by Heart Disease Status", fig)


def chart_pairplot(train):
    df = train[["age", "chol", "thalach", "oldpeak", "target"]]
    g = sns.pairplot(df, hue="target", palette=["#C87DDE", "#9820D4"], diag_kind="kde", corner=False)
    g.figure.suptitle("Pairwise Relationships Between Key Features", y=1.02, color="white", fontsize=15)
    g.figure.set_facecolor("#1a0b2e")
    for ax in g.axes.flatten():
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#4c2a6b")
    g._legend.get_frame().set_facecolor("#2a1647")
    g._legend.get_frame().set_edgecolor("#4c2a6b")
    for text in g._legend.get_texts():
        text.set_color("white")
    return _chart("Pairwise Relationships Between Key Features", g.figure)


def chart_cv_boxplot(train_fe, target):
    import lightgbm

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVC": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "LightGBM": lightgbm.LGBMClassifier(random_state=42, verbosity=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_df = pd.DataFrame(
        {
            name: cross_val_score(model, train_fe, target, cv=cv, scoring="accuracy")
            for name, model in models.items()
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#1a0b2e")
    sns.boxplot(data=cv_df, palette="RdPu", ax=ax)
    sns.stripplot(data=cv_df, color="white", alpha=0.5, size=4, ax=ax)
    ax.set_title("5-Fold Cross-Validation Accuracy by Model", color="white", fontsize=15)
    ax.set_ylabel("Accuracy", color="white")
    ax.tick_params(colors="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, color="white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("5-Fold Cross-Validation Accuracy by Model", fig)


def chart_confusion_matrix(test_fe, target, best_name="LightGBM (Exported)"):
    model = joblib.load(MODEL_PATH)
    preds = model.predict(test_fe)
    cm = confusion_matrix(target, preds)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#1a0b2e")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdPu", cbar=False,
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {best_name}", color="white", fontsize=14)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.tick_params(colors="white")
    fig.tight_layout()
    return _chart("Confusion Matrix", fig)


def chart_roc_curve(test_fe, target):
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(test_fe)[:, 1]
    fpr, tpr, _ = roc_curve(target, proba)
    auc = roc_auc_score(target, proba)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a0b2e")
    ax.plot(fpr, tpr, color=PURPLE, linewidth=2.5, label=f"LightGBM (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9f86c0", label="Random Guess")
    ax.set_title("ROC Curve", color="white", fontsize=15)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.tick_params(colors="white")
    ax.legend(loc="lower right", facecolor="#2a1647", edgecolor="#4c2a6b")
    for text in ax.legend_.get_texts():
        text.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("ROC Curve", fig)


def chart_feature_importance(feature_names):
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "feature_importances_"):
        return None
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="#1a0b2e")
    importances.plot(
        kind="barh",
        color=plt.cm.RdPu(np.linspace(0.4, 0.9, len(importances))),
        ax=ax,
    )
    ax.set_title("Feature Importance — LightGBM", color="white", fontsize=15)
    ax.set_xlabel("Importance", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("Feature Importance", fig)


# ---------------------------------------------------- new-factor charts -----

def chart_weight(train):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a0b2e")
    sns.histplot(
        data=train, x="weight", hue="target", kde=True, multiple="stack",
        palette="RdPu", bins=25, ax=ax,
    )
    ax.set_title("Weight Distribution by Heart Disease Status", color="white", fontsize=15)
    ax.set_xlabel("Weight (kg)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    ax.legend(labels=["No Disease", "Disease"], facecolor="#2a1647", edgecolor="#4c2a6b")
    for text in ax.legend_.get_texts():
        text.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")
    fig.tight_layout()
    return _chart("Weight Distribution by Heart Disease Status", fig)


def _style_axis(ax):
    ax.tick_params(colors="white")
    if ax.legend_ is not None:
        ax.legend_.get_frame().set_facecolor("#2a1647")
        ax.legend_.get_frame().set_edgecolor("#4c2a6b")
        for text in ax.legend_.get_texts():
            text.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#4c2a6b")


def _bar_with_rate(fig_ax, train, col, labels):
    fig, ax = fig_ax
    sns.countplot(data=train, x=col, hue="target", palette="RdPu", ax=ax)
    ax.set_title(f"Heart Disease Count by {col.title()} Status", color="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, color="white")
    ax.set_xlabel(col.title(), color="white")
    ax.set_ylabel("Count", color="white")
    ax.legend(labels=["No Disease", "Disease"], facecolor="#2a1647", edgecolor="#4c2a6b")
    for text in ax.legend_.get_texts():
        text.set_color("white")
    _style_axis(ax)
    return fig


def chart_smoking(train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a0b2e")
    _bar_with_rate((fig, axes[0]), train, "smoking", ["Non-Smoker", "Smoker"])
    sns.barplot(data=train, x="smoking", y="target", hue="smoking", palette=PURPLES[:2], legend=False, ax=axes[1])
    axes[1].set_title("Heart Disease Rate by Smoking Status", color="white")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Non-Smoker", "Smoker"], color="white")
    axes[1].set_xlabel("Smoking", color="white")
    axes[1].set_ylabel("Disease Rate", color="white")
    _style_axis(axes[1])
    fig.tight_layout()
    return _chart("Smoking vs Heart Disease", fig)


def chart_diabetes(train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a0b2e")
    _bar_with_rate((fig, axes[0]), train, "diabetes", ["No Diabetes", "Diabetes"])
    sns.barplot(data=train, x="diabetes", y="target", hue="diabetes", palette=PURPLES[:2], legend=False, ax=axes[1])
    axes[1].set_title("Heart Disease Rate by Diabetes Status", color="white")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["No Diabetes", "Diabetes"], color="white")
    axes[1].set_xlabel("Diabetes", color="white")
    axes[1].set_ylabel("Disease Rate", color="white")
    _style_axis(axes[1])
    fig.tight_layout()
    return _chart("Diabetes vs Heart Disease", fig)


def generate_graphs():
    """Return a list of {"title": str, "image": data-uri} for every chart."""
    train, test = _load_data()
    charts = [
        chart_numeric_distributions(train),
        chart_target_count(train),
        chart_target_pie(train),
        chart_gender_bar(train),
        chart_gender_cp(train),
        chart_cp_rate(train),
        chart_age_chol(train),
        chart_age_hist(train),
        chart_weight(train),
        chart_smoking(train),
        chart_diabetes(train),
        chart_thalach_violin(train),
        chart_boxplots(train),
        chart_corr_heatmap(train),
        chart_pairplot(train),
    ]

    train_fe = _encode(train)
    test_fe = _encode(test)
    feature_names = list(train_fe.drop(columns=["target"]).columns)
    if "target" in test_fe.columns:
        test_target = test_fe["target"].fillna(0).astype(int)
    else:
        test_target = pd.Series(np.zeros(len(test_fe)), index=test_fe.index)

    train_target = train_fe["target"]
    charts.append(chart_cv_boxplot(train_fe[feature_names], train_target))

    test_fe = test_fe[feature_names]
    cm_chart = chart_confusion_matrix(test_fe, test_target)
    if cm_chart:
        charts.append(cm_chart)
    roc_chart = chart_roc_curve(test_fe, test_target)
    if roc_chart:
        charts.append(roc_chart)
    imp_chart = chart_feature_importance(feature_names)
    if imp_chart:
        charts.append(imp_chart)

    return charts


if __name__ == "__main__":
    result = generate_graphs()
    print(f"Generated {len(result)} charts")
    for c in result:
        print(f"- {c['title']} ({len(c['image']) // 1024} KB)")
