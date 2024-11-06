# DetectVideoShotLength

`DetectVideoShotLength` is a Python package for detecting scene cuts in videos using advanced histogram and structural similarity (SSIM) analysis. This tool identifies frames where abrupt scene changes (or "cuts") occur, saves cut frames, and provides insights into shot durations, distribution, and visual frequency.

It is ideal for projects requiring video analysis, such as filmmaking, video editing, or computer vision research.

## Key Features
- **Cut Detection**: Detects scene cuts based on RGB histogram changes and verifies cuts using SSIM thresholds.
- **Scene Duration Analysis**: Calculates the duration of each scene and provides insights into average shot lengths.
- **Scene Length Visualization**: Plots scene lengths over time, showing the pacing and rhythm of a video.
- **Scene Frequency Distribution**: Visualizes the distribution of scene lengths in a logarithmic frequency chart.
- **Sorted Scene Listing**: Lists scenes sorted by duration, with start and end timestamps for detailed scene timing.


## Implementation

Change `threshold_hist` from `0` to `1`, the default threshold would be `0.15`
and `threshold_ssim` from `0.75` to `0.95`, default would be `0.85`

Usually, the parameters give different results with day scenes and night scenes.

If the video is large, a segment could be analyzed.

    import DetectVideoShotLength

    video_path = r"D:\___Research\average shot\test_video.mp4"
    output_dir = r"D:\___Research\average shot\video"
    
    results = DetectVideoShotLength.detect_cuts(video_path, output_dir, threshold_hist=0.20, threshold_ssim=0.80, min_minutes=5, max_minutes=7)

    DetectVideoShotLength.plot_scene_lengths(results)
    DetectVideoShotLength.list_scenes_sorted_by_length(results)
    DetectVideoShotLength.plot_scene_length_frequencies(results)
