import pyautogui
import time
from typing import List, Dict
from PIL import Image
import os
from rich import print
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..models.element_extractor import ElementExtractor
from ..utils.vision import get_som_labeled_img, check_ocr_box

class ComputerController:
    def __init__(self, 
                 som_model,
                 caption_model_processor,
                 device: str = 'cuda',
                 image_folder: str = 'imgs'):
        """Initialize the computer controller."""
        self.som_model = som_model
        self.caption_model_processor = caption_model_processor
        self.device = device
        self.image_folder = image_folder
        self._setup_llm()
        self._ensure_image_folder()
        
    def _setup_llm(self):
        """Setup the language model for element extraction."""
        system = """You are a great element name extractor from a text. 
        You can find the name of the element/tool/page/any button that the user is asking you to click or go to.
        For example: 'Can you go to start', here you will find 'start' as the name of the element or 'Can you open app folder.', you will see 'app'.
        Give a single word or just 2-3 word outputs."""
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.structured_llm_grader = llm.with_structured_output(ElementExtractor)
        
        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "This is the content of the whole Desktop page {page_elements}.\n\n User question: {question}"),
        ])
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader
        
    def _ensure_image_folder(self):
        """Ensure the image folder exists."""
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
            
    def _take_screenshot(self) -> str:
        """Take a screenshot and save it."""
        timestamp = int(time.time())
        image_path = os.path.join(self.image_folder, f"screenshot_{timestamp}.png")
        screenshot = pyautogui.screenshot()
        screenshot.save(image_path)
        return image_path
        
    def process_screenshot(self, image_path: str):
        """Process a screenshot and return the labeled image with coordinates."""
        image = Image.open(image_path)
        image_rgb = image.convert('RGB')
        box_overlay_ratio = image.size[0] / 3200
        
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        ocr_bbox_rslt, _ = check_ocr_box(
            image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )
        
        text, ocr_bbox = ocr_bbox_rslt
        return get_som_labeled_img(
            image_path, 
            self.som_model,
            BOX_TRESHOLD=0.03,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.1,
            imgsz=640
        )
        
    def run(self):
        """Run the computer controller."""
        try:
            print("\n=== Starting Computer Controller ===")
            while True:
                print("\n=== New Iteration Started ===")
                
                # Take screenshot
                print("📸 Taking screenshot...")
                image_path = self._take_screenshot()
                print(f"   Screenshot saved to: {image_path}")
                
                # Process screenshot
                print("\n🔍 Processing screenshot...")
                labeled_img, label_coordinates, parsed_content_list = self.process_screenshot(image_path)
                print("   Found elements:")
                for idx, content in enumerate(parsed_content_list):
                    print(f"   {idx}: {content}")
                print("\n📍 Element coordinates:")
                for idx, coords in label_coordinates.items():
                    print(f"   Element {idx}: (x={coords[0]:.2f}, y={coords[1]:.2f})")
                
                # Clean up old screenshot
                print("\n🗑️  Cleaning up screenshot...")
                os.remove(image_path)
                
                # Clean the content list
                print("\n🧹 Cleaning content list...")
                cleaned_content_list = [item.split(": ", 1)[1] if ": " in item else item for item in parsed_content_list]
                print("   Cleaned elements:")
                for idx, content in enumerate(cleaned_content_list):
                    print(f"   {idx}: {content}")
                
                # Ask for user input
                print("\n❓ Waiting for user input...")
                question = input("Ask a question (or type 'exit' to stop): ")
                if question.lower() == "exit":
                    print("\n👋 Exiting program...")
                    break
                    
                print("\n🤖 Processing question through LLM...")
                print(f"   Question: {question}")
                print("   Available elements:", cleaned_content_list)
                response = self.retrieval_grader.invoke({
                    "question": question, 
                    "page_elements": cleaned_content_list
                })
                target_content = response.binary_score
                print(f"   LLM identified target: '{target_content}'")
                
                # Move and click
                print("\n🖱️  Attempting to move and click...")
                self._move_and_click(parsed_content_list, label_coordinates, target_content)
                
                # Sleep for the configured interval
                print("\n⏳ Waiting for next iteration...")
                time.sleep(1.0)  # TODO: Use config.screenshot_interval
                
        except KeyboardInterrupt:
            print("\n⛔ Process interrupted by user. Stopping computer control system...")
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
            raise
            
    def _move_and_click(self, parsed_content_list, label_coordinates, target_content):
        """Move to and click on the target content."""
        print(f"   🔎 Searching for content: '{target_content}'")
        
        # Find the content index
        index = None
        for idx, content in enumerate(parsed_content_list):
            print(f"   Checking element {idx}: '{content}'")
            if target_content in content:
                index = idx
                print(f"   ✅ Found match at index {idx}")
                break
                
        if index is None:
            print(f"   ❌ Content '{target_content}' not found in any elements")
            return
            
        # Get the coordinates
        print(f"   📍 Getting coordinates for index {index}")
        coordinates = label_coordinates.get(str(index))
        if coordinates is None:
            print(f"   ❌ No coordinates found for index {index}")
            return
            
        # Extract x, y from coordinates and move the cursor
        x, y = coordinates[0], coordinates[1]
        print(f"   🖱️  Moving cursor to coordinates: ({x:.2f}, {y:.2f})")
        pyautogui.moveTo(x, y, duration=0.5)  # Smoothly move the cursor
        print("   🖱️  Clicking...")
        pyautogui.click()  # Perform a click action
        print(f"   ✅ Successfully clicked on '{target_content}' at coordinates: ({x:.2f}, {y:.2f})") 