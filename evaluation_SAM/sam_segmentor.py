from ultralytics import SAM
import torch
import os


def segment(input_image_folder: str, output_file_path: str):
    """
    Segment all images in the input folder using SAM segmentor and save results to output file.
    Args:
        input_image_folder (str): Folder containing input images.
        output_file_path (str): Path to save the segmentation results. Should end with '.pt'.
    Returns:
        all_model_results (list): List of dictionaries with model names and their segmentation results.
    """

    # Define models to test (just provide the weights file or identifier)
    model_list = [
        #{"name": "YOLOv11n", "type": "YOLO", "weights": "yolo11n-seg.pt"}, #bad
        #{"name": "YOLOv11x", "type": "YOLO", "weights": "yolo11x-seg.pt"}, #bad
        #{"name": "SAM_b", "type": "SAM", "weights": "sam_b.pt"}, #sam_l better
        {"name": "SAM_l", "type": "SAM", "weights": "sam_l.pt"},
        {"name": "SAM2.1_b", "type": "SAM", "weights": "sam2.1_b.pt"}
    ]

    # Load all image paths recursively
    if not os.path.exists(input_image_folder):
        raise ValueError(f"Image folder '{input_image_folder}' does not exist.")

    image_paths = []
    for root, _, files in os.walk(input_image_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))


    all_model_results = []
    # Loop through each model
    for model_config in model_list:
        print(f"\nRunning inference with: {model_config['name']}")

        # Load model, run inference
        if model_config["type"] == "YOLO":
            model = YOLO(model_config["weights"])
            results = model(image_paths)

        elif model_config["type"] == "SAM":
            model = SAM(model_config["weights"])
            results = []
            for image in image_paths:
                results.append(model(image))

        else:
            raise ValueError("Unsupported model type.")

        # save results
        all_model_results.append({
            "model_name": model_config["name"],
            "results": results
        })

        # print 
        print(f"Runned inference with {model_config['name']} on {len(results)} images.")


    # save all_model_results to output file
    if not output_file_path.endswith('.pt'):
        raise ValueError("Output file must end with .pt")
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
        
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    torch.save(all_model_results, output_file_path)
    print(f"Saved segmentation results to: {output_file_path}")

    return all_model_results