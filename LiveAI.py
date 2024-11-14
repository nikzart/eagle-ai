import cv2
import openai
import base64
import threading
import queue
import time
from openai import AzureOpenAI
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from tiktoken import encoding_for_model
import json
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv

class VideoAnalyzer:
    def __init__(self, context_window=30):  # context_window in seconds
        """
        Initializes the VideoAnalyzer with Azure OpenAI configuration.
        """
        # Load environment variables
        load_dotenv()
        
        self.client = AzureOpenAI(
            api_key=os.getenv('AZURE_API_KEY'),
            api_version=os.getenv('AZURE_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_ENDPOINT')
        )
        
        self.context_window = context_window
        self.frame_buffer = deque(maxlen=context_window * 5)  # Store 5 frames per second
        self.analysis = ""
        self.lock = threading.Lock()
        self.target_size = (320, 240)
        self.context_history = deque(maxlen=10)  # Store last 10 analyses
        
        # Add pricing constants (in INR)
        self.INPUT_PRICE_PER_TOKEN = 0.165 * 84.40 / 1_000_000  # INR per token
        self.OUTPUT_PRICE_PER_TOKEN = 0.66 * 84.40 / 1_000_000  # INR per token
        self.VISION_PRICE_PER_IMAGE = 0.001275 * 84.40  # INR per image
        
        # Add token counters
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_images = 0
        self.tokenizer = encoding_for_model("gpt-4")

    def encode_frame(self, frame):
        """Encode the frame as JPEG and convert to base64."""
        resized = cv2.resize(frame, self.target_size)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', rgb_frame, encode_param)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_as_text}"

    def get_context_summary(self):
        """Get a summary of the recent context history."""
        if not self.context_history:
            return "No previous context available."
        return " | ".join(self.context_history)

    def analyze_frames(self, frame_data):
        """
        Send the current frame and context to Azure OpenAI GPT-4 for analysis.
        """
        try:
            # Get the latest frame and context
            latest_frame = frame_data['encoded_frame']
            context = self.get_context_summary()
            
            # Count input tokens
            system_message = "You are an assistant that analyzes video streams. Consider the context of the last 30 seconds when analyzing. Provide brief, concise descriptions of what's happening."
            user_message = f"Previous context: {context}\n\nWhat's happening now? Describe in 10 words or less."
            
            input_tokens = len(self.tokenizer.encode(system_message)) + len(self.tokenizer.encode(user_message))
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": latest_frame,
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            analysis = response.choices[0].message.content.strip()
            output_tokens = len(self.tokenizer.encode(analysis))
            
            with self.lock:
                self.analysis = analysis
                self.context_history.append(analysis)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_images += 1
                
        except Exception as e:
            print(f"Error during Azure OpenAI API call: {e}")
            with self.lock:
                self.analysis = "Analysis unavailable."

    def start_analysis_thread(self):
        """Starts a separate thread to handle frame analysis."""
        threading.Thread(target=self.analysis_worker, daemon=True).start()

    def analysis_worker(self):
        """Worker function that continuously processes frames with context."""
        while True:
            try:
                # Get the latest frame data
                if len(self.frame_buffer) > 0:
                    latest_frame = self.frame_buffer[-1]
                    self.analyze_frames(latest_frame)
                time.sleep(0.2)  # Analysis every 200ms
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(1.0)

    def update_analysis(self, frame):
        """Adds a frame to the buffer with timestamp."""
        try:
            encoded = self.encode_frame(frame)
            frame_data = {
                'timestamp': time.time(),
                'encoded_frame': encoded
            }
            self.frame_buffer.append(frame_data)
            
            # Clean up old frames
            current_time = time.time()
            while (len(self.frame_buffer) > 0 and 
                   current_time - self.frame_buffer[0]['timestamp'] > self.context_window):
                self.frame_buffer.popleft()
                
        except Exception as e:
            print(f"Update error: {e}")

    def get_analysis(self):
        """Retrieves the latest analysis."""
        with self.lock:
            return self.analysis

    def get_cost_analysis(self):
        """Calculate and return the current cost analysis."""
        input_cost = self.total_input_tokens * self.INPUT_PRICE_PER_TOKEN
        output_cost = self.total_output_tokens * self.OUTPUT_PRICE_PER_TOKEN
        vision_cost = self.total_images * self.VISION_PRICE_PER_IMAGE
        total_cost = input_cost + output_cost + vision_cost
        
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'images': self.total_images,
            'total_cost_inr': total_cost
        }

def cv2_frame_to_pil(frame):
    """Convert CV2 frame to PIL Image"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_cv2_frame(pil_image):
    """Convert PIL Image to CV2 frame"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def draw_text_with_pil(frame, text_elements):
    """Draw text elements using PIL with proper Rupee symbol and text wrapping"""
    pil_image = cv2_frame_to_pil(frame)
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    def wrap_text(text, max_width):
        """Helper function to wrap text"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            # Check if current line width exceeds max_width
            line_width = draw.textlength(" ".join(current_line), font=font)
            if line_width > max_width:
                if len(current_line) == 1:
                    # Word itself is too long, just add it
                    lines.append(current_line[0])
                    current_line = []
                else:
                    # Remove last word and add line
                    current_line.pop()
                    lines.append(" ".join(current_line))
                    current_line = [word]
        
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    # Calculate maximum width for text (leave margin on both sides)
    max_width = pil_image.width - 20

    # Draw analysis text at the top with wrapping
    y_position = 10
    if 'analysis' in text_elements:
        analysis_text = text_elements['analysis']
        wrapped_lines = wrap_text(analysis_text, max_width)
        for line in wrapped_lines:
            draw.text((10, y_position), line, fill=(0, 255, 0), font=font)
            y_position += 30

    # Draw cost and time at the bottom
    if 'cost' in text_elements:
        # Wrap cost text if needed
        cost_lines = wrap_text(text_elements['cost'], max_width)
        cost_y = pil_image.height - 50 - (len(cost_lines) - 1) * 25
        for line in cost_lines:
            draw.text((10, cost_y), line, fill=(255, 255, 0), font=font)
            cost_y += 25
    
    if 'time' in text_elements:
        draw.text((10, pil_image.height - 25), text_elements['time'], 
                 fill=(255, 255, 0), font=font)

    return pil_to_cv2_frame(pil_image)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    analyzer = VideoAnalyzer(context_window=30)
    analyzer.start_analysis_thread()

    frame_count = 0
    last_analysis_time = time.time()
    start_time = time.time()
    analysis_interval = 0.2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        current_time = time.time()
        elapsed_time = current_time - start_time

        # Update analysis if enough time has passed
        if current_time - last_analysis_time >= analysis_interval:
            analyzer.update_analysis(frame)
            last_analysis_time = current_time

        # Display the analysis with text wrapping
        analysis = analyzer.get_analysis()
        if analysis:
            # Get cost analysis
            cost_info = analyzer.get_cost_analysis()
            
            # Format elapsed time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            
            # Prepare text elements
            text_elements = {
                'analysis': f"Analysis: {analysis}",
                'cost': f"Tokens: {cost_info['input_tokens']}in/{cost_info['output_tokens']}out | Images: {cost_info['images']} | Cost: ₹{cost_info['total_cost_inr']:.2f}",
                'time': f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
            }
            
            # Draw text using PIL
            frame = draw_text_with_pil(frame, text_elements)

        cv2.imshow('Webcam Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print final statistics
    final_costs = analyzer.get_cost_analysis()
    final_elapsed = time.time() - start_time
    hours = int(final_elapsed // 3600)
    minutes = int((final_elapsed % 3600) // 60)
    seconds = int(final_elapsed % 60)
    
    print("\nFinal Usage Statistics:")
    print(f"Total Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Input Tokens: {final_costs['input_tokens']}")
    print(f"Output Tokens: {final_costs['output_tokens']}")
    print(f"Images Processed: {final_costs['images']}")
    print(f"Total Cost: ₹{final_costs['total_cost_inr']:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()