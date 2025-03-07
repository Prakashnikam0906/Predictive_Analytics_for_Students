from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_model(X_train, y_train):
    """ Train a Random Forest model and save it. """
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save the trained model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully!")
    return model  # ✅ Returning the trained model
