from sklearn.tree import DecisionTreeRegressor

from misc import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    cross_validate_model
)


def main():

    print("=" * 50)
    print("Decision Tree Regressor")
    print("=" * 50)

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=5
    )

    cv_mse = cross_validate_model(
        model,
        X_train,
        y_train
    )

    trained_model = train_model(
        model,
        X_train,
        y_train
    )

    test_mse = evaluate_model(
        trained_model,
        X_test,
        y_test
    )

    print(f"Cross Validation MSE : {cv_mse:.4f}")
    print(f"Test Set MSE         : {test_mse:.4f}")


if __name__ == "__main__":
    main()
