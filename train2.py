from sklearn.kernel_ridge import KernelRidge

from misc import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    cross_validate_model
)


def main():

    print("=" * 50)
    print("Kernel Ridge Regressor")
    print("=" * 50)

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = KernelRidge(
        alpha=1.0,
        kernel="rbf",
        gamma=0.1
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
