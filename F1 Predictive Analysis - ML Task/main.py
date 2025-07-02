import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("f1_pitstops_2018_2024.csv")

# Preserve driver→code mapping
driver_le = LabelEncoder()
df['DriverCode'] = driver_le.fit_transform(df['Driver'])

# Points assignment
points_map = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
df['Points'] = df['Position'].map(points_map).fillna(0)

# Identify season champions
points_df = df.groupby(['Season','DriverCode'])['Points'].sum().reset_index()
champions = points_df.loc[points_df.groupby('Season')['Points'].idxmax()]
df['IsChampion'] = 0
for _,row in champions.iterrows():
    mask = (df['Season']==row['Season']) & (df['DriverCode']==row['DriverCode'])
    df.loc[mask,'IsChampion'] = 1

# Drop columns (errors='ignore' skips any missing names)
df.drop(columns=[
    'Driver','Points',
    'Race Name','Date','Time_of_race','Location','Country',
    'Abbreviation','Pit_Lap','Circuit'
], inplace=True, errors='ignore')

# Drop rows with too many NAs
df.dropna(thresh=20, inplace=True)

# Fill numeric and categorical missing values
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Label-encode all remaining object cols
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Aggregate per season/driver
agg = df.groupby(['Season','DriverCode']).agg({
    'Constructor':'first',
    'Position':'mean',
    'TotalPitStops':'mean',
    'AvgPitStopTime':'mean',
    'Air_Temp_C':'mean',
    'Track_Temp_C':'mean',
    'Humidity_%':'mean',
    'Wind_Speed_KMH':'mean',
    'Lap Time Variation':'mean',
    'Tire Usage Aggression':'mean',
    'Fast Lap Attempts':'mean',
    'Position Changes':'mean',
    'Driver Aggression Score':'mean',
    'Stint':'mean',
    'Tire Compound':'mean',
    'Stint Length':'mean',
    'Laps':'sum',
    'IsChampion':'max'
}).reset_index()

agg.columns = [
    'Season','DriverCode','Constructor','AvgPosition','AvgTotalPitStops',
    'AvgPitStopTime','AvgAirTemp','AvgTrackTemp','AvgHumidity','AvgWindSpeed',
    'AvgLapTimeVar','AvgTireAggression','AvgFastLapAttempts','AvgPosChanges',
    'AvgDriverAggression','AvgStint','AvgTireCompound','AvgStintLength',
    'TotalLaps','IsChampion'
]

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(agg.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Feature importance up to 2023
X_all = agg[agg['Season']<=2023].drop(['Season','DriverCode','IsChampion'],axis=1)
y_all = agg[agg['Season']<=2023]['IsChampion']
rf_all = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_all.fit(X_all, y_all)
pd.Series(rf_all.feature_importances_, index=X_all.columns)\
  .nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

# Train/test split
train = agg[agg['Season']<=2022]
test  = agg[agg['Season']==2023]
X_train = train.drop(['Season','DriverCode','IsChampion'],axis=1)
y_train = train['IsChampion']
X_test  = test.drop( ['Season','DriverCode','IsChampion'],axis=1)
y_test  = test['IsChampion']

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(class_weight='balanced'),
    'Naive Bayes':         GaussianNB(),
    'KNN':                 KNeighborsClassifier()
}

for name,mdl in models.items():
    mdl.fit(X_train,y_train)
    y_pred = mdl.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("F1 Score:",f1_score(y_test,y_pred,average='weighted'))
    print(classification_report(y_test,y_pred))
    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.show()

# Predict 2024 with RF
best_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
best_rf.fit(X_train,y_train)
df2024 = agg[agg['Season']==2024].copy()
X_2024 = df2024.drop(['Season','DriverCode','IsChampion'],axis=1)
df2024['ChampionProbability'] = best_rf.predict_proba(X_2024)[:,1]

# Reverse map driver codes → names
df2024['Driver'] = driver_le.inverse_transform(df2024['DriverCode'].astype(int))

# Show top 5 and champion
top5 = df2024[['Driver','ChampionProbability']]\
           .sort_values('ChampionProbability',ascending=False).head()
print("Top 5 Predicted 2024 Drivers:\n", top5)
print("\nPredicted 2024 Champion:", top5.iloc[0]['Driver'])
