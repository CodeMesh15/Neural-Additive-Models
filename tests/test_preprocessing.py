import pandas as pd

def test_no_missing_in_cleaned_data():
    df = pd.read_csv("data/X_train.csv")
    assert not df.isnull().values.any(), "No NaNs should remain in cleaned data."
    print("Preprocessing test passed.")

if __name__ == "__main__":
    test_no_missing_in_cleaned_data()
