import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("f1_pitstops_2018_2024.csv")
driver_le = LabelEncoder()
df['DriverCode'] = driver_le.fit_transform(df['Driver'])
df['Points'] = df['Position'].map({1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}).fillna(0)
season_pts = df.groupby(['Season','DriverCode'])['Points'].sum().reset_index()
champions = season_pts.loc[season_pts.groupby('Season')['Points'].idxmax()]
df['IsChampion'] = 0
for _, r in champions.iterrows():
    mask = (df['Season']==r['Season']) & (df['DriverCode']==r['DriverCode'])
    df.loc[mask, 'IsChampion'] = 1

df.drop(columns=['Driver','Points','Race Name','Date','Time_of_race','Location','Country','Abbreviation','Pit_Lap','Circuit'], inplace=True, errors='ignore')
df.dropna(thresh=20, inplace=True)
num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode().iloc[0])
    df[col] = LabelEncoder().fit_transform(df[col])

agg = df.groupby(['Season','DriverCode']).agg({
    'Constructor': lambda x: x.mode().iloc[0],'Position':'mean','TotalPitStops':'mean','AvgPitStopTime':'mean',
    'Air_Temp_C':'mean','Track_Temp_C':'mean','Humidity_%':'mean','Wind_Speed_KMH':'mean',
    'Lap Time Variation':'mean','Tire Usage Aggression':'mean','Fast Lap Attempts':'mean',
    'Position Changes':'mean','Driver Aggression Score':'mean','Stint':'mean',
    'Tire Compound':'mean','Stint Length':'mean','Laps':'sum','IsChampion':'max'
}).reset_index()
agg.columns = ['Season','DriverCode','Constructor','AvgPosition','AvgTotalPitStops','AvgPitStopTime','AvgAirTemp','AvgTrackTemp','AvgHumidity','AvgWindSpeed','AvgLapTimeVar','AvgTireAggression','AvgFastLapAttempts','AvgPosChanges','AvgDriverAggression','AvgStint','AvgTireCompound','AvgStintLength','TotalLaps','IsChampion']

plt.figure(figsize=(12,8))
sns.heatmap(agg.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

X_all = agg[agg['Season'] <= 2023].drop(['Season','DriverCode','IsChampion'], axis=1)
y_all = agg[agg['Season'] <= 2023]['IsChampion']
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_all, y_all)
rf_temp = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_temp.fit(X_all, y_all)
importances = pd.Series(rf_temp.feature_importances_, index=X_all.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

feat = agg.copy()
feat['FeatureSeason'] = feat['Season']
feat['LabelSeason'] = feat['Season'] + 1
lbl = agg[['Season','DriverCode','IsChampion']].rename(columns={'Season':'LabelSeason','IsChampion':'ChampionNext'})
df_ml = feat.merge(lbl, on=['LabelSeason','DriverCode'], how='inner')
df_ml.drop(columns=['Season','LabelSeason','IsChampion'], inplace=True)

train_ml = df_ml[df_ml['FeatureSeason'] <= 2022]
test_ml = df_ml[df_ml['FeatureSeason'] == 2023]
X_tr = train_ml.drop(['FeatureSeason','DriverCode','ChampionNext'], axis=1)
y_tr = train_ml['ChampionNext']
X_te = test_ml.drop(['FeatureSeason','DriverCode','ChampionNext'], axis=1)
y_te = test_ml['ChampionNext']
ros = RandomOverSampler(random_state=42)
X_tr_res, y_tr_res = ros.fit_resample(X_tr, y_tr)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', max_depth=5, random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_tr_res, y_tr_res)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average='weighted')
    results[name] = {'model': model, 'accuracy': acc, 'f1': f1}
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f} — Weighted F1: {f1:.3f}")
    print(classification_report(y_te, y_pred, digits=3))
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

best_name = max(results, key=lambda n: results[n]['f1'])
best_model = results[best_name]['model']
print(f"\n--> Best model: {best_name} (F1 = {results[best_name]['f1']:.3f})")

final_train = df_ml[df_ml['FeatureSeason'] <= 2023]
X_ft = final_train.drop(['FeatureSeason','DriverCode','ChampionNext'], axis=1)
y_ft = final_train['ChampionNext']
X_ft_res, y_ft_res = ros.fit_resample(X_ft, y_ft)
best_model.fit(X_ft_res, y_ft_res)

df_2024 = agg[agg['Season']==2024].copy()
X_2024 = df_2024.drop(['Season','DriverCode','IsChampion'], axis=1)
df_2024['Champion2025Prob'] = best_model.predict_proba(X_2024)[:,1]
df_2024['Driver'] = driver_le.inverse_transform(df_2024['DriverCode'].astype(int))
top5 = df_2024[['Driver','Champion2025Prob']].sort_values('Champion2025Prob', ascending=False).head(5)
print(f"\nTop 5 Predicted 2025 Drivers:")
print(top5.to_string(index=False))
print(f"\nPredicted 2025 Champion: {top5.iloc[0]['Driver']}")
