import model
import torch

def predict(input_data: torch.Tensor, output_number: int, model_path: str) -> torch.Tensor:
    assert input_data.dtype == torch.float32, "Input data must be of type torch.float32"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the model
    model_load = model.ANN().to(device)
    input_data = input_data.to(device)
    model_load.load_state_dict(torch.load(model_path))
    model_load.eval()   
    
    # Predict
    with torch.no_grad():
        output = model_load(input_data)

    # Top output
    result = output.topk(output_number, dim=1).indices
    return result


