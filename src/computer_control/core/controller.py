"""Computer controller module."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pyautogui
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image
from rich.console import Console
from ultralytics import YOLO

from ..models.element_extractor import ElementExtractor
from ..utils.vision import check_ocr_box, get_som_labeled_img


class ComputerController:
    def __init__(
        self,
        som_model: YOLO,
        caption_model_processor: Dict[str, Any],
        device: str = "cuda",
        image_folder: str = "imgs",
    ) -> None:
        """Initialize the computer controller."""
        self.som_model = som_model
        self.caption_model_processor = caption_model_processor
        self.device = device
        self.image_folder = image_folder
        self.console = Console()
        self._setup_llm()
        self._ensure_image_folder()

    def _setup_llm(self) -> None:
        """Setup the language model for element extraction."""
        system = (
            "You are a great element name extractor from a text.\n"
            "You can find the name of the element/tool/page/any button that "
            "the user is asking you to click or go to.\n"
            "For example: 'Can you go to start', here you will find 'start' "
            "as the name of the element or 'Can you open app folder.', "
            "you will see 'app'.\n"
            "Give a single word or just 2-3 word outputs."
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.structured_llm_grader = llm.with_structured_output(ElementExtractor)

        prompt_template = (
            "This is the content of the whole Desktop page {page_elements}.\n\n"
            "User question: {question}"
        )
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", prompt_template),
            ]
        )
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader

    def _ensure_image_folder(self) -> None:
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

    def process_screenshot(
        self, image_path: str
    ) -> Tuple[Image.Image, Dict[str, List[float]], List[str]]:
        """Process a screenshot and return the labeled image with coordinates."""
        image = Image.open(image_path)
        box_overlay_ratio = image.size[0] / 3200

        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }

        ocr_bbox_rslt, _ = check_ocr_box(
            image_path,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": 0.9},
            use_paddleocr=True,
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
            imgsz=640,
        )

    def run(self) -> None:
        """Run the computer controller."""
        try:
            self.console.print("\n[bold blue]=== Starting Computer Controller ===[/bold blue]")
            while True:
                self.console.print("\n[bold cyan]=== New Iteration Started ===[/bold cyan]")

                # Take screenshot
                self.console.print("\n[bold green]📸 Taking screenshot...[/bold green]")
                image_path = self._take_screenshot()
                self.console.print(f"   Screenshot saved to: [italic]{image_path}[/italic]")

                # Process screenshot
                self.console.print("\n[bold yellow]🔍 Processing screenshot...[/bold yellow]")
                result = self.process_screenshot(image_path)
                labeled_img, label_coordinates, parsed_content_list = result

                self.console.print("[bold]   Found elements:[/bold]")
                for idx, content in enumerate(parsed_content_list):
                    self.console.print(f"   {idx}: [cyan]{content}[/cyan]")

                self.console.print("\n[bold]📍 Element coordinates:[/bold]")
                for key_idx, coords in label_coordinates.items():
                    coord_str = f"(x={coords[0]:.2f}, y={coords[1]:.2f})"
                    self.console.print(f"   Element {key_idx}: [yellow]{coord_str}[/yellow]")

                # Clean up old screenshot
                self.console.print("\n[bold magenta]🗑️  Cleaning up screenshot...[/bold magenta]")
                os.remove(image_path)

                # Clean the content list
                self.console.print("\n[bold]🧹 Cleaning content list...[/bold]")
                cleaned_content_list = [
                    item.split(": ", 1)[1] if ": " in item else item for item in parsed_content_list
                ]
                self.console.print("[bold]   Cleaned elements:[/bold]")
                for idx, content in enumerate(cleaned_content_list):
                    self.console.print(f"   {idx}: [cyan]{content}[/cyan]")

                # Ask for user input
                self.console.print("\n[bold green]❓ Waiting for user input...[/bold green]")
                question = input("Ask a question (or type 'exit' to stop): ")
                if question.lower() == "exit":
                    self.console.print("\n[bold red]👋 Exiting program...[/bold red]")
                    break

                self.console.print(
                    "\n[bold purple]🤖 Processing question through LLM...[/bold purple]"
                )
                self.console.print(f"   Question: [italic]{question}[/italic]")
                self.console.print("   Available elements:", cleaned_content_list)
                response = self.retrieval_grader.invoke(
                    {"question": question, "page_elements": cleaned_content_list}
                )
                if not isinstance(response, ElementExtractor):
                    raise TypeError("Unexpected response type from LLM")
                target_content = response.binary_score
                self.console.print(
                    f"   LLM identified target: [bold green]'{target_content}'[/bold green]"
                )

                # Move and click
                self.console.print(
                    "\n[bold yellow]🖱️  Attempting to move and click...[/bold yellow]"
                )
                self._move_and_click(parsed_content_list, label_coordinates, target_content)

                # Sleep for the configured interval
                self.console.print("\n[bold blue]⏳ Waiting for next iteration...[/bold blue]")
                time.sleep(1.0)  # TODO: Use config.screenshot_interval

        except KeyboardInterrupt:
            msg = "Process interrupted by user. Stopping computer control system..."
            self.console.print(f"\n[bold red]⛔ {msg}[/bold red]")
        except Exception as e:
            self.console.print(f"\n[bold red]❌ Error occurred: {str(e)}[/bold red]")
            raise

    def _move_and_click(
        self,
        parsed_content_list: List[str],
        label_coordinates: Dict[str, List[float]],
        target_content: str,
    ) -> None:
        """Move to and click on the target content."""
        self.console.print(
            f"   [bold]🔎 Searching for content: [cyan]'{target_content}'[/cyan][/bold]"
        )

        # Find the content index
        found_idx: Optional[int] = None
        for idx, content in enumerate(parsed_content_list):
            self.console.print(f"   Checking element {idx}: [italic]'{content}'[/italic]")
            if target_content in content:
                found_idx = idx
                self.console.print(f"   [bold green]✅ Found match at index {idx}[/bold green]")
                break

        if found_idx is None:
            self.console.print(
                f"   [bold red]❌ Content '{target_content}' not found in any elements[/bold red]"
            )
            return

        # Get the coordinates using string key
        coord_key = str(found_idx)
        self.console.print(f"   [bold]📍 Getting coordinates for index {coord_key}[/bold]")
        coordinates = label_coordinates.get(coord_key)
        if coordinates is None:
            self.console.print(
                f"   [bold red]❌ No coordinates found for index {coord_key}[/bold red]"
            )
            return

        # Extract x, y from coordinates and move the cursor
        x, y = coordinates[0], coordinates[1]
        self.console.print(
            f"   [bold yellow]🖱️  Moving cursor to coordinates: ({x:.2f}, {y:.2f})[/bold yellow]"
        )
        pyautogui.moveTo(x, y, duration=0.5)  # Smoothly move the cursor
        self.console.print("   [bold yellow]🖱️  Clicking...[/bold yellow]")
        pyautogui.click()  # Perform a click action
        coord_str = f"({x:.2f}, {y:.2f})"
        msg = f"Successfully clicked on '{target_content}' at coordinates: {coord_str}"
        self.console.print(f"   [bold green]✅ {msg}[/bold green]")
