import pandas as pd
import matplotlib.pyplot as plt

def extract_metrics(df, model_name):
    tn, fp = df.iloc[0]
    fn, tp = df.iloc[1]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    }

# Dosyaları oku ve temizle
auto_df = pd.read_csv("reports/autoencoder_confusion_matrix.csv").iloc[:, 1:3]
iso_df = pd.read_csv("reports/isolation_forest_confusion_matrix.csv").iloc[:, 1:3]
xgb_df = pd.read_csv("reports/xgboost_confusion_matrix.csv").iloc[:, 1:3]

# Hesapla
results = [
    extract_metrics(auto_df, "Autoencoder"),
    extract_metrics(iso_df, "Isolation Forest"),
    extract_metrics(xgb_df, "XGBoost")
]

df_results = pd.DataFrame(results)
print("\n📊 Model Performance Comparison:\n")
print(df_results)

# CSV olarak kaydet
df_results.to_csv("reports/model_comparison_summary.csv", index=False)

#  Görselleştirme
models = df_results["Model"]
accuracy = df_results["Accuracy"]
precision = df_results["Precision"]
recall = df_results["Recall"]
f1_score = df_results["F1-Score"]

bar_width = 0.2
x = range(len(models))

plt.figure(figsize=(12, 6))
plt.bar([i - 1.5*bar_width for i in x], accuracy, width=bar_width, label="Accuracy")
plt.bar([i - 0.5*bar_width for i in x], precision, width=bar_width, label="Precision")
plt.bar([i + 0.5*bar_width for i in x], recall, width=bar_width, label="Recall")
plt.bar([i + 1.5*bar_width for i in x], f1_score, width=bar_width, label="F1-Score")

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.xticks(ticks=x, labels=models)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Görseli kaydet
plt.savefig("reports/model_comparison_plot.png")
plt.show()
