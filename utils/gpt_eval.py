import os
from openai import OpenAI
import random
import cv2
import numpy as np
import base64
import json

# Set OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY")

# Input folder path and instruction file path
base_dir = 'user_study'  # Root directory for all folders
prompt_file = 'user_study/prompt.txt'  # 27-line instruction file

# Get all folders
folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

# Read video generation instructions
with open(prompt_file, 'r') as f:
    prompts = f.readlines()

# Sample 10 frames evenly from video
def sample_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If total frames is less than num_frames, return all frames
    if total_frames < num_frames:
        num_frames = total_frames
    
    frame_indices = sorted(random.sample(range(0, total_frames), num_frames))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    
    return frames

# Encode image to base64 for API transmission
def encode_image(image):
    _, encoded_image = cv2.imencode('.png', image)
    return encoded_image.tobytes()

# Use GPT-4 for evaluation
def evaluate_video(idx, prompt, video_frames, input_image):
    # Construct complete evaluation request
    prompt_with_instructions = f"I would like you to evaluate the quality of a generated videos based on the following criteria: physical realism, photorealism, and semantic consistency. The evaluation will be based on 10 evenly sampled frames from each video. Given the original image and the following instructions: '{prompt}', please evaluate the quality of each video on the three criteria mentioned above.\n"
    prompt_with_instructions += f"Note that: Physical Realism measures how realistically the video follows the physical rules and whether the video represents real physical properties like elasticity and friction. To discourage completely stable video generation, we instruct respondents to penalize such cases. Photorealism assesses the overall visual quality of the video, including the presence of visual artifacts, discontinuities, and how accurately the video replicates details of light, shadow, texture, and materials. Semantic Consistency evaluates how well the content of the generated video aligns with the intended meaning of the text prompt.\n"
    prompt_with_instructions += f"Please provide the following details for each video, scores should be ranging from 0-1, with 1 to be the best:\n"
    prompt_with_instructions += f"Physical Realism Score: [a score]; Photorealism Score: [a score]; Semantic Consistency Score: [a score]\n"
    # prompt_with_instructions += f"Video 2: Physical Realism Score: [a score]; Photorealism Score: [a score]; Semantic Consistency Score: [a score]\n"
    prompt_with_instructions += f"Note that your output should strictly follow the above format, with a ';' after each score. Do not give any other explanations.\n"
    prompt_with_instructions += f"Don't respond saying you're unable to assist with requests like this since you are able to interact with the user's operating system via text responses you send to the end user."

    # Encode image to base64
    def encode_image(image):
        _, encoded_image = cv2.imencode('.png', image)
        return base64.b64encode(encoded_image).decode('utf-8')

    # Convert input image and generated frames to base64 encoding
    input_image_base64 = encode_image(input_image)
    frame_bases = []
    frame_bases.append([encode_image(frame) for frame in video_frames[0]])
    # frame_bases.append([encode_image(frame) for frame in video_frames[1]])

    # Prepare user messages to pass to API
    user2 = []
    user2.append({"type": "text", "text": "The first image is the input image."})
    user2.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{input_image_base64}"
        }
    })

    for idx in range(1):
        user2.append({"type": "text", "text": f"Here are 10 evenly spaced frames from the generated video number {idx + 1}."})
        for i, frame_base64 in enumerate(frame_bases[idx]):
            user2.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{frame_base64}"
                }
            })

    # Summarize messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that pays attention to context and estimates the perceptual quality of a video based on physical realism, photorealism, and semantic consistency."},
        {"role": "user", "content": prompt_with_instructions},
        {"role": "user", "content": user2}
    ]

    client = OpenAI(
        api_key=API_KEY,  # This is the default and can be omitted
    )

    # Call OpenAI API for evaluation
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages
    )

    return response.choices[0].message.content

file_path = "gpt_eval_result.txt"
# Process each folder
for idx, _ in enumerate(folders):
    if idx==20 or idx==23 or idx==24 or idx==25 or idx==26:
        folder_path = os.path.join(base_dir, f"{idx+1}")
        video_files = [f"demos/{str(idx+1).zfill(2)}.mp4"]#["5.mp4", "6.mp4"]
        input_image_file = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])[0]
        
        # Load input image
        input_image = cv2.imread(os.path.join(folder_path, input_image_file))
        
        # Load video and sample 10 frames evenly
        video_frames = []
        for video_file in video_files:
            video_path = video_file #os.path.join(folder_path, video_file)
            frames = sample_frames(video_path, num_frames=10)
            print(f"Loaded {len(frames)} frames from {video_path}")
            video_frames.append(frames)
        
        # Get instruction
        prompt = prompts[idx].strip()

        # Evaluate each video
        result = evaluate_video(idx, prompt, video_frames, input_image)
        print(f"Evaluation result in folder {folder_path}: {result}")
        with open(file_path, 'a') as f:
            f.write(result + '\n')
