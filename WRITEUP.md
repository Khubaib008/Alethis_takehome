# Write-up
## 1. Your timings (100 words)
How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time?
The initial version was giving 30 fps on same hardware (on colab). However, by using the final optimized version gave approximately 185 fps. This was mainly achieved as we batched INT8 Tensor RT and also parallelize things, meaning, having multiple instaces running in parallel. 


## 2. Your pipeline (150 words)
Describe what your script does, step by step, from opening a Parquet file to writing an output line.
a reader thread reads parquet files into a shared memory. decoder worker decode, resize, rotate, gpu instances run inference, adn result stored as per instructions.


## 3. Where the time goes (150 words)
Which part of your final version takes the most time? How did you measure this?
GPU acted as abottleneck and utilization of decoder and gpu was measured using CUDA events.

## 4. Checking correctness (100 words)
How did you check that your results stayed correct after you made the script faster?
comparing final results with the reference results obtained using running it using default settings and storing those results as ground truth.


## 5. Working with AI (100 words)
Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it?
AI suggested to skip frames for faster inference but i rejected that idea and opted for processing alll frames.


## 6. Next steps (100 words)
If you had one more week, what would you change, and why?
further optimize and try to test hardware to max and see how many instances can run efficiently and testing on it. Further understanding the model architecture, understanding which operations are cpu optimized and which are gpu optimized and make pipeline accordingly. Also trying to allocate more resources to bottleneck.

