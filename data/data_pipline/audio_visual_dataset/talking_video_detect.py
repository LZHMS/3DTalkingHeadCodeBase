"""
Talking Video Detection Module

This module detects and validates talking videos by checking if frames contain 
faces that meet size requirements. It can filter out invalid videos and move them 
to a separate directory.

Main Features:
    1. Face detection and size validation
    2. Smart frame sampling strategies
    3. Video duration filtering
    4. Automatic invalid video removal
    5. Metadata update after filtering
"""

# Import everything, just for illustration purposes
import cv2
import numpy as np
from ibug.face_detection import RetinaFacePredictor
from ibug.face_detection.utils import HeadPoseEstimator, SimpleFaceTracker
from typing import Tuple, List, Optional
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class TalkingVideoDetector:
    """
    Detector for identifying valid talking videos with properly sized faces.
    
    This class uses RetinaFace for face detection and validates that detected faces
    meet minimum size requirements relative to the frame dimensions.
    """
    
    def __init__(self, 
                 face_size_threshold: float = 0.1,
                 confidence_threshold: float = 0.8,
                 device: str = 'cuda:0'):
        """
        Initialize the talking video detector.
        
        Args:
            face_size_threshold (float): Minimum face area ratio relative to frame (0-1)
            confidence_threshold (float): Face detection confidence threshold
            device (str): Device to use ('cuda:0' or 'cpu')
        """
        self.face_size_threshold = face_size_threshold
        self.confidence_threshold = confidence_threshold
        
        # Create RetinaFace detector
        self.face_detector = RetinaFacePredictor(
            threshold=confidence_threshold, 
            device=device,
            model=RetinaFacePredictor.get_model('resnet50'))
        
        # Create head pose estimator
        self.pose_estimator = HeadPoseEstimator()
        
        # Create face tracker
        self.face_tracker = SimpleFaceTracker(minimum_face_size=64)
    
    def calculate_face_ratio(self, face_box: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        """
        Calculate the ratio of face area to frame area.
        
        Args:
            face_box (np.ndarray): Face bounding box [left, top, right, bottom]
            frame_shape (Tuple[int, int]): Frame dimensions (height, width)
        
        Returns:
            float: Ratio of face area to total frame area
        """
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_height * frame_width
        
        face_width = face_box[2] - face_box[0]
        face_height = face_box[3] - face_box[1]
        face_area = face_width * face_height
        
        return face_area / frame_area
    
    def detect_face_in_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Detect whether a frame contains a valid face meeting requirements.
        
        Args:
            frame (np.ndarray): Video frame in BGR format
        
        Returns:
            Tuple[bool, Optional[np.ndarray], Optional[float]]: 
                - Whether a valid face is detected
                - Face detection result
                - Face area ratio
        """
        # Detect faces
        detected_faces = self.face_detector(frame, rgb=False)
        
        if len(detected_faces) == 0:
            return False, None, 0.0
        
        # Find the largest face
        max_face_idx = 0
        max_face_ratio = 0.0
        
        for idx, face in enumerate(detected_faces):
            face_ratio = self.calculate_face_ratio(face[:4], frame.shape)
            if face_ratio > max_face_ratio:
                max_face_ratio = face_ratio
                max_face_idx = idx
        
        # Check if the largest face meets the threshold
        if max_face_ratio >= self.face_size_threshold:
            return True, detected_faces[max_face_idx], max_face_ratio
        else:
            return False, detected_faces[max_face_idx], max_face_ratio
    
    def get_smart_sample_indices(self, total_frames: int, 
                                 strategy: str = 'adaptive',
                                 min_samples: int = 30,
                                 max_samples: int = 300) -> List[int]:
        """
        Generate smart sampling frame indices.
        
        Args:
            total_frames (int): Total number of frames in video
            strategy (str): Sampling strategy
                - 'three_frames': Sample only first, middle, and last frames (fastest)
                - 'uniform': Uniform sampling
                - 'adaptive': Adaptive sampling (more samples for short videos, fewer for long)
                - 'keyframe': Keyframe sampling (dense sampling at start, middle, end)
            min_samples (int): Minimum number of samples
            max_samples (int): Maximum number of samples
        
        Returns:
            List[int]: List of frame indices to sample
        """
        if strategy == 'three_frames':
            # Sample only first, middle, and last frames
            first_frame = 0
            middle_frame = total_frames // 2
            last_frame = total_frames - 1
            indices = [first_frame, middle_frame, last_frame]
        
        elif strategy == 'uniform':
            # Uniform sampling
            num_samples = min(max_samples, max(min_samples, total_frames // 10))
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
        
        elif strategy == 'adaptive':
            # Adaptive sampling: dynamically adjust based on video length
            if total_frames <= 300:  # Short video (10 seconds @30fps)
                num_samples = min(total_frames, max(min_samples, total_frames // 3))
            elif total_frames <= 1800:  # Medium video (1 minute @30fps)
                num_samples = min(max_samples, max(min_samples, total_frames // 10))
            else:  # Long video
                num_samples = min(max_samples, max(min_samples, total_frames // 20))
            
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
        
        elif strategy == 'keyframe':
            # Keyframe sampling: dense sampling at start, end, and middle
            # First 20% dense sampling
            start_count = max(10, int(min_samples * 0.3))
            start_indices = np.linspace(0, int(total_frames * 0.2), start_count, dtype=int)
            
            # Middle 60% sparse sampling
            middle_count = max(10, int(min_samples * 0.4))
            middle_indices = np.linspace(int(total_frames * 0.2), int(total_frames * 0.8), 
                                        middle_count, dtype=int)
            
            # Last 20% dense sampling
            end_count = max(10, int(min_samples * 0.3))
            end_indices = np.linspace(int(total_frames * 0.8), total_frames - 1, 
                                     end_count, dtype=int)
            
            indices = np.concatenate([start_indices, middle_indices, end_indices])
            indices = np.unique(indices).tolist()
            
            # Limit maximum sample count
            if len(indices) > max_samples:
                step = len(indices) // max_samples
                indices = indices[::step]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        return sorted(set(indices))
    
    def check_video(self, video_path: str, 
                    sampling_strategy: str = 'adaptive',
                    min_samples: int = 30,
                    max_samples: int = 300,
                    verbose: bool = True) -> dict:
        """
        Check if all sampled frames in the video contain valid talking faces.
        
        Args:
            video_path (str): Path to video file
            sampling_strategy (str): Sampling strategy ('uniform', 'adaptive', 'keyframe')
            min_samples (int): Minimum number of samples
            max_samples (int): Maximum number of samples
            verbose (bool): Whether to print detailed information
        
        Returns:
            dict: Detection results containing validation status and statistics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Generate sampling indices
        sample_indices = self.get_smart_sample_indices(
            total_frames, 
            strategy=sampling_strategy,
            min_samples=min_samples,
            max_samples=max_samples
        )
        
        if verbose:
            print(f"Total frames: {total_frames}, Sampled frames: {len(sample_indices)}, "
                  f"Sampling rate: {len(sample_indices)/total_frames*100:.1f}%")
        
        valid_frames = 0
        invalid_frames = 0
        face_ratios = []
        invalid_frame_indices = []
        
        # Only read sampled frames
        for idx, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                invalid_frames += 1
                invalid_frame_indices.append(frame_idx)
                continue
            
            has_valid_face, face_data, face_ratio = self.detect_face_in_frame(frame)
            
            if has_valid_face:
                valid_frames += 1
                face_ratios.append(face_ratio)
                
                if verbose and (idx + 1) % 50 == 0:
                    print(f"Checked {idx + 1}/{len(sample_indices)} frames, "
                          f"face ratio {face_ratio:.3f}")
            else:
                invalid_frames += 1
                invalid_frame_indices.append(frame_idx)
                
                if verbose:
                    print(f"Frame {frame_idx}: Invalid "
                          f"(face ratio {face_ratio:.3f}, threshold {self.face_size_threshold})")
        
        cap.release()
        
        # Calculate statistics
        avg_face_ratio = np.mean(face_ratios) if face_ratios else 0.0
        min_face_ratio = np.min(face_ratios) if face_ratios else 0.0
        max_face_ratio = np.max(face_ratios) if face_ratios else 0.0
        
        result = {
            'is_valid': valid_frames > 0,
            'total_frames': total_frames,
            'sampled_frames': len(sample_indices),
            'sampling_rate': len(sample_indices) / total_frames,
            'valid_frames': valid_frames,
            'invalid_frames': invalid_frames,
            'avg_face_ratio': avg_face_ratio,
            'min_face_ratio': min_face_ratio,
            'max_face_ratio': max_face_ratio,
            'invalid_frame_indices': invalid_frame_indices,
            'fps': fps,
            'sampling_strategy': sampling_strategy
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Detection results:")
            print(f"  Total frames: {total_frames}")
            print(f"  Sampled frames: {len(sample_indices)} ({len(sample_indices)/total_frames*100:.1f}%)")
            print(f"  Sampling strategy: {sampling_strategy}")
            print(f"  Valid frames: {valid_frames}")
            print(f"  Invalid frames: {invalid_frames}")
            print(f"  Average face ratio: {avg_face_ratio:.3f}")
            print(f"  Minimum face ratio: {min_face_ratio:.3f}")
            print(f"  Maximum face ratio: {max_face_ratio:.3f}")
            print(f"  Video valid: {'Yes' if result['is_valid'] else 'No'}")
            print(f"{'='*60}")
        
        return result


def batch_check_videos(video_paths: List[str],
                       face_size_threshold: float = 0.1,
                       sampling_strategy: str = 'adaptive',
                       device: str = 'cuda:0') -> dict:
    """
    Batch check multiple videos.
    
    Args:
        video_paths (List[str]): List of video file paths
        face_size_threshold (float): Face area ratio threshold
        sampling_strategy (str): Sampling strategy
        device (str): Device to use
    
    Returns:
        dict: Detection results for each video
    """
    detector = TalkingVideoDetector(
        face_size_threshold=face_size_threshold,
        device=device
    )
    
    results = {}
    for video_path in video_paths:
        print(f"\n{'='*60}")
        print(f"Checking video: {video_path}")
        print(f"{'='*60}")
        
        try:
            result = detector.check_video(
                video_path, 
                sampling_strategy=sampling_strategy,
                verbose=True
            )
            results[video_path] = result
        except Exception as e:
            print(f"Failed to check video: {e}")
            results[video_path] = {'error': str(e)}
    
    return results


def process_output_clips(base_dir: str = 'output_clips',
                        output_removed_dir: str = 'output_removed',
                        face_size_threshold: float = 0.02,
                        min_duration: float = 5.0,
                        device: str = 'cuda:0'):
    """
    Process output_clips directory, detect faces in videos, and remove invalid videos.
    
    Invalid videos are those that either:
        1. Have no valid faces in sampled frames
        2. Have duration less than min_duration seconds
    
    Args:
        base_dir (str): Path to output_clips directory
        output_removed_dir (str): Target directory for removed files
        face_size_threshold (float): Face area ratio threshold
        min_duration (float): Minimum video duration in seconds (default: 5.0)
        device (str): Device to use ('cuda:0' or 'cpu')
    """
    import json
    import shutil
    
    # Create detector
    detector = TalkingVideoDetector(
        face_size_threshold=face_size_threshold,
        confidence_threshold=0.8,
        device=device
    )
    
    # Traverse category directories
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory does not exist")
        return
    
    categories = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    total_videos = 0
    removed_videos = 0
    
    for category in categories:
        category_path = os.path.join(base_dir, category)
        print(f"\n{'='*80}")
        print(f"Processing category: {category}")
        print(f"{'='*80}")
        
        # Traverse video ID directories
        video_ids = [d for d in os.listdir(category_path) 
                     if os.path.isdir(os.path.join(category_path, d))]
        
        for video_id in video_ids:
            video_id_path = os.path.join(category_path, video_id)
            print(f"\nProcessing video ID: {video_id}")
            
            # Get all mp4 files
            mp4_files = [f for f in os.listdir(video_id_path) 
                        if f.endswith('.mp4')]
            
            removed_scenes = []
            
            for mp4_file in mp4_files:
                total_videos += 1
                video_path = os.path.join(video_id_path, mp4_file)
                
                # Get scene ID (e.g., scene_8.mp4 -> scene_8)
                scene_id = os.path.splitext(mp4_file)[0]
                
                print(f"  Detecting video: {mp4_file}")
                
                try:
                    # Check video duration first
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    
                    # Check if video is too short
                    if duration < min_duration:
                        print(f"    ✗ Video too short: {duration:.2f}s < {min_duration}s")
                        cap.release()
                        
                        # Move video and audio files
                        removed_videos += 1
                        removed_scenes.append(scene_id)
                        
                        # Build target path
                        rel_path = os.path.relpath(video_id_path, base_dir)
                        target_dir = os.path.join(output_removed_dir, rel_path)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Move mp4 file
                        target_video = os.path.join(target_dir, mp4_file)
                        shutil.move(video_path, target_video)
                        print(f"    Moved: {video_path} -> {target_video}")
                        
                        # Move corresponding m4a file
                        audio_file = scene_id + '.m4a'
                        audio_path = os.path.join(video_id_path, audio_file)
                        if os.path.exists(audio_path):
                            target_audio = os.path.join(target_dir, audio_file)
                            shutil.move(audio_path, target_audio)
                            print(f"    Moved: {audio_path} -> {target_audio}")
                        
                        continue
                    
                    # Uniformly sample 5 frames for face detection
                    sample_indices = np.linspace(0, total_frames - 1, 5, dtype=int).tolist()
                    
                    # Detect faces in sampled frames
                    has_valid_face_found = False
                    
                    for frame_idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        
                        if not ret:
                            continue
                        
                        has_valid_face, _, face_ratio = detector.detect_face_in_frame(frame)
                        
                        if has_valid_face:
                            has_valid_face_found = True
                            print(f"    ✓ Frame {frame_idx}: Valid face detected (ratio: {face_ratio:.4f})")
                            break
                    
                    cap.release()
                    
                    if not has_valid_face_found:
                        # Move video and audio files
                        removed_videos += 1
                        removed_scenes.append(scene_id)
                        
                        # Build target path
                        rel_path = os.path.relpath(video_id_path, base_dir)
                        target_dir = os.path.join(output_removed_dir, rel_path)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Move mp4 file
                        target_video = os.path.join(target_dir, mp4_file)
                        shutil.move(video_path, target_video)
                        print(f"    Moved: {video_path} -> {target_video}")
                        
                        # Move corresponding m4a file
                        audio_file = scene_id + '.m4a'
                        audio_path = os.path.join(video_id_path, audio_file)
                        if os.path.exists(audio_path):
                            target_audio = os.path.join(target_dir, audio_file)
                            shutil.move(audio_path, target_audio)
                            print(f"    Moved: {audio_path} -> {target_audio}")
                    else:
                        print(f"    ✓ Video valid")
                
                except Exception as e:
                    print(f"    Error: {e}")
            
            # Update scene_info.json and video_scece_info.txt
            if removed_scenes:
                print(f"\n  Updating metadata files, removing scenes: {removed_scenes}")
                
                # Update scene_info.json
                scene_info_path = os.path.join(video_id_path, 'scene_info.json')
                if os.path.exists(scene_info_path):
                    try:
                        with open(scene_info_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        # Filter out removed scenes
                        filtered_lines = []
                        for line in lines:
                            try:
                                data = json.loads(line.strip())
                                if data.get('id') not in removed_scenes:
                                    filtered_lines.append(line)
                            except:
                                pass
                        
                        # Write back to file
                        with open(scene_info_path, 'w', encoding='utf-8') as f:
                            f.writelines(filtered_lines)
                        
                        print(f"  Updated: scene_info.json")
                    except Exception as e:
                        print(f"  Failed to update scene_info.json: {e}")
                
                # Update video_scece_info.txt
                video_info_path = os.path.join(video_id_path, 'video_scece_info.txt')
                if os.path.exists(video_info_path):
                    try:
                        with open(video_info_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        # Filter out removed scenes
                        filtered_lines = []
                        for line in lines:
                            # Check if line contains removed scene ID
                            # Format: scene X infos: ...
                            should_keep = True
                            for scene_id in removed_scenes:
                                # Extract scene number (e.g., scene_8 -> 8)
                                scene_num = scene_id.split('_')[-1]
                                if f'scene {scene_num} ' in line.lower():
                                    should_keep = False
                                    break
                            
                            if should_keep:
                                filtered_lines.append(line)
                        
                        # Write back to file
                        with open(video_info_path, 'w', encoding='utf-8') as f:
                            f.writelines(filtered_lines)
                        
                        print(f"  Updated: video_scece_info.txt")
                    except Exception as e:
                        print(f"  Failed to update video_scece_info.txt: {e}")
    
    print(f"\n{'='*80}")
    print(f"Processing completed!")
    print(f"Total videos: {total_videos}")
    print(f"Removed videos: {removed_videos}")
    print(f"Kept videos: {total_videos - removed_videos}")
    print(f"{'='*80}")


# Usage example
if __name__ == "__main__":
    # Process output_clips directory
    process_output_clips(
        base_dir='output_clips',
        output_removed_dir='output_removed',
        face_size_threshold=0.02,
        min_duration=5.0,
        device='cuda:0'
    )