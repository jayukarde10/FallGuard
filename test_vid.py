import os, sys, tempfile, imageio_ffmpeg, subprocess
sys.path.append(os.getcwd() + '/aimlproject')
from video_pipeline import VideoProcessor
from c3d_model import load_c3d_model
from face_blur import FaceAnonymizer
from dataset_manager import GMDCSADataset

ds = GMDCSADataset('ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76')
meta = ds.get_video_record('Subject 1', 'Fall', '01.mp4')

print(f"Input: {meta['video_path']}")

c3d = load_c3d_model('weights.h5')
fa = FaceAnonymizer('models')
vp = VideoProcessor(c3d, 'models/fall_head.joblib', fa)

out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
res = vp.process_video(meta['video_path'], out, True, 'gaussian', 45, 0.25, 0.5, 0.5, 8, lambda c,t,p: None)

print('Out exists?', os.path.exists(out))
print('Out size:', os.path.getsize(out))

exe = imageio_ffmpeg.get_ffmpeg_exe()
print('--- FFprobe of generated processed video ---')
subprocess.run([exe, '-i', out])
