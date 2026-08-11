model = {
    "name": "fraud-detector",
    "version": 1,
    "framework": "scikit-learn",
    "accuracy": 0.94,
    "features": [
        "transaction_amount",
        "transaction_frequency",
        "location"
    ]
}

print(f"Model Name: {model['name']}")
print(f"Version: {model['version']}")
print(f"Framework: {model['framework']}")
print(f"Accuracy: {model['accuracy']:.1%}")
print(f"Number of Features: {len(model['features'])}")