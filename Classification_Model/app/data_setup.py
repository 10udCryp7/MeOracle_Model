import torch
def transform(symptoms: list) -> torch.Tensor:
    # Create matrix of zeros
    data = [[0] * 377]
    # Set the symptoms to 1
    for symptom in symptoms:
        data[0][symptom] = 1
    return torch.tensor(data, dtype=torch.float32)


