import os
import json
import torch
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torch.cuda.amp import autocast


def post_process_boxes(boxes, scores, img_w, img_h):
    if len(boxes) == 0:
        return [], []

    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    min_area = 0.001 * img_w * img_h
    max_area = 0.5 * img_w * img_h
    size_keep = (areas > min_area) & (areas < max_area)

    boxes = boxes[size_keep]
    scores = scores[size_keep]

    if len(boxes) == 0:
        return [], []

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    aspect = torch.maximum(w / h, h / w)
    aspect_keep = aspect > 2.0

    boxes = boxes[aspect_keep]
    scores = scores[aspect_keep]

    if len(boxes) == 0:
        return [], []

    score_keep = scores > 0.3
    boxes = boxes[score_keep]
    scores = scores[score_keep]

    if len(boxes) == 0 or scores.max() < 0.4:
        return [], []

    keep = nms(boxes, scores, 0.5)
    boxes = boxes[keep]
    scores = scores[keep]

    final_keep = scores > 0.6
    boxes = boxes[final_keep]
    scores = scores[final_keep]

    return boxes.tolist(), scores.tolist()


def run_inference(
    image_path,
    model,
    processor,
    save_path,
    prompts,
    box_threshold,
    text_threshold,
    visualize_results,
    visualization_path,
    device,
    batch_size=1
):
    image_names = sorted(os.listdir(image_path))
    rows = []

    model.eval()

    for i in tqdm(range(0, len(image_names), batch_size)):
        batch_names = image_names[i:i + batch_size]
        images = [
            Image.open(os.path.join(image_path, n)).convert("RGB")
            for n in batch_names
        ]

        all_boxes = [[] for _ in images]
        all_scores = [[] for _ in images]

        for prompt in prompts:
            inputs = processor(
                images=images,
                text=[prompt],
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad(), autocast():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[img.size[::-1] for img in images]
            )

            for idx, result in enumerate(results):
                for box, score in zip(result["boxes"], result["scores"]):
                    all_boxes[idx].append(box.tolist())
                    all_scores[idx].append(score.item())

        for img, name, boxes, scores in zip(images, batch_names, all_boxes, all_scores):
            boxes, scores = post_process_boxes(
                boxes, scores, img.width, img.height
            )

            coco_boxes = []
            category_ids = []

            for box in boxes:
                xmin, ymin, xmax, ymax = box
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img.width, xmax)
                ymax = min(img.height, ymax)
                coco_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                category_ids.append(0)

            if visualize_results:
                draw = ImageDraw.Draw(img)
                for box in boxes:
                    draw.rectangle(box, outline="red", width=3)
                img.save(os.path.join(visualization_path, name))

            rows.append({
                "file_name": name,
                "bbox": json.dumps(coco_boxes),
                "category_id": json.dumps(category_ids)
            })

    pd.DataFrame(rows).to_csv(save_path, index=False)


if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    current_directory = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE_PATH = "/tmp/data/test_images"
    SUBMISSION_SAVE_PATH = os.path.join(current_directory, "submission.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(os.path.join(current_directory, "processor"))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        os.path.join(current_directory, "model")
    ).to(device)

    BOX_THRESHOLD = 0.2
    TEXT_THRESHOLD = 0.2

    PROMPTS = [
        "a surgical instrument",
        "Large Needle Driver",
        "Prograsp Forceps",
        "Monopolar Curved Scissors",
        "metal surgical tool"
    ]

    parent_directory = os.path.dirname(current_directory)
    visualization_path = os.path.join(parent_directory, "outputs")
    visualize_results = False

    if visualize_results:
        os.makedirs(visualization_path, exist_ok=True)

    run_inference(
        TEST_IMAGE_PATH,
        model,
        processor,
        SUBMISSION_SAVE_PATH,
        PROMPTS,
        BOX_THRESHOLD,
        TEXT_THRESHOLD,
        visualize_results,
        visualization_path,
        device
    )
