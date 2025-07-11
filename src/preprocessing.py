import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path):
    # Veriyi oku
    df = pd.read_csv(path)

    # 'Amount' ve 'Time' değişkenlerini ölçekle
    scaler = StandardScaler()
    df[['scaled_amount', 'scaled_time']] = scaler.fit_transform(df[['Amount', 'Time']])

    # Kullanılmayacak orijinal sütunları kaldır
    df = df.drop(['Amount', 'Time'], axis=1)

    # scaled sütunları başa al
    cols = ['scaled_amount', 'scaled_time'] + [col for col in df.columns if col not in ['scaled_amount', 'scaled_time', 'Class']] + ['Class']
    df = df[cols]

    # Özellikler ve etiket ayrımı
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Eğitim ve test ayırımı (Stratify = sınıf dağılımı korunur)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
