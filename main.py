from src.data_loader import load_data
from src.model_trainer import train_model
from src.model_evaluator import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Filepath for dataset
DATA_PATH = "data/synthetic_student_performance.csv"

# Load data
df = load_data(DATA_PATH)

if df is not None:
    # Encode categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    # Split data into features (X) and target variable (y)
    X = df.drop(columns=['G3'])
    y = df['G3']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model ✅ (Now it works!)
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
