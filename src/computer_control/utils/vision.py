from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
import torch
from paddleocr import PaddleOCR
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from ultralytics import YOLO


def get_yolo_model(model_path: str) -> YOLO:
    """Load and return a YOLO model."""
    model = YOLO(model_path)
    return model

def get_caption_model_processor(
    model_name: str, 
    model_name_or_path: str, 
    device: str
) -> Dict[str, Any]:
    """Load and return a caption model processor."""
    processor = Blip2Processor.from_pretrained(model_name_or_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    return {"processor": processor, "model": model}

def check_ocr_box(
    image_path: str,
    display_img: bool = False,
    output_bb_format: str = 'xyxy',
    goal_filtering: Any = None,
    easyocr_args: Optional[Dict[str, Any]] = None,
    use_paddleocr: bool = True
) -> Tuple[Tuple[List[str], List[List[float]]], bool]:
    """Check OCR box in the image."""
    if use_paddleocr:
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ([], []), False
            
        texts = []
        boxes = []
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            if goal_filtering is None or goal_filtering(text):
                texts.append(text)
                # Convert box format if needed
                if output_bb_format == 'xyxy':
                    x1, y1 = box[0]
                    x2, y2 = box[2]
                    boxes.append([x1, y1, x2, y2])
                else:
                    boxes.append(box)
    else:
        reader = easyocr.Reader(['en'])
        easyocr_args = easyocr_args or {}
        result = reader.readtext(image_path, **easyocr_args)
        
        texts = []
        boxes = []
        for detection in result:
            box = detection[0]
            text = detection[1]
            if goal_filtering is None or goal_filtering(text):
                texts.append(text)
                if output_bb_format == 'xyxy':
                    x1, y1 = box[0]
                    x2, y2 = box[2]
                    boxes.append([x1, y1, x2, y2])
                else:
                    boxes.append(box)
                    
    return (texts, boxes), True

def get_som_labeled_img(
    image_path: str,
    som_model: YOLO,
    BOX_TRESHOLD: float = 0.03,
    output_coord_in_ratio: bool = False,
    ocr_bbox: Optional[List[List[float]]] = None,
    draw_bbox_config: Optional[Dict[str, Any]] = None,
    caption_model_processor: Optional[Dict[str, Any]] = None,
    ocr_text: Optional[List[str]] = None,
    use_local_semantics: bool = True,
    iou_threshold: float = 0.1,
    imgsz: int = 640
) -> Tuple[Image.Image, Dict[str, List[float]], List[str]]:
    """Get SOM labeled image with coordinates and content."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    image_np = np.array(image)
    
    # Run YOLO detection
    results = som_model(image_np, imgsz=imgsz)[0]
    
    # Process detections
    boxes = []
    scores = []
    for box in results.boxes:
        if box.conf.item() > BOX_TRESHOLD:
            boxes.append(box.xyxy[0].tolist())
            scores.append(box.conf.item())
    
    # Process OCR results if available
    if ocr_bbox and ocr_text:
        for bbox, _text in zip(ocr_bbox, ocr_text, strict=False):
            boxes.append(bbox)
            scores.append(1.0)  # High confidence for OCR results
    
    # Generate captions for detected regions
    captions = []
    label_coordinates = {}

    if caption_model_processor is None:
        raise ValueError("caption_model_processor cannot be None")

    processor = caption_model_processor["processor"]
    model = caption_model_processor["model"]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        region = image_np[y1:y2, x1:x2]
        
        # Convert region to PIL Image
        region_pil = Image.fromarray(region)
        
        # Get caption using BLIP-2
        inputs = processor(
            images=region_pil, 
            return_tensors="pt"
        ).to(model.device)
        
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=5
        )
        caption = processor.batch_decode(
            out, 
            skip_special_tokens=True
        )[0]
        
        captions.append(caption)
        
        # Store coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if output_coord_in_ratio:
            center_x /= image_np.shape[1]
            center_y /= image_np.shape[0]
        label_coordinates[str(i)] = [center_x, center_y]
    
    # Draw boxes and captions on image
    image_with_boxes = image_np.copy()
    bbox_config = draw_bbox_config or {
        'thickness': 2,
        'text_scale': 0.5,
        'text_thickness': 1
    }
    
    for _i, (box, caption) in enumerate(zip(boxes, captions, strict=False)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            image_with_boxes,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=bbox_config.get('thickness', 2)
        )
        cv2.putText(
            image_with_boxes,
            caption[:30],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            bbox_config.get('text_scale', 0.5),
            (0, 255, 0),
            bbox_config.get('text_thickness', 1)
        )
    
    return Image.fromarray(image_with_boxes), label_coordinates, captions 