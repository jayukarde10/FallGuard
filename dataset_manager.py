import os
import re
import glob
import pandas as pd

def parse_intervals(class_str):
    """
    Parses class interval strings like:
    'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]'
    Returns a list of dicts: [{'action': 'Sitting', 'start': 0.0, 'end': 3.4, 'is_fall': False}, ...]
    """
    results = []
    if not isinstance(class_str, str):
        return results

    parts = [p.strip() for p in class_str.split(';') if p.strip()]
    for p in parts:
        # Match pattern: ActionName [start to end] or ActionName[start to end]
        m = re.match(r'^(.*?)\s*\[\s*([\d\.]+)\s+to\s+([\d\.]+)\s*\]', p)
        if m:
            act = m.group(1).strip()
            try:
                start = float(m.group(2))
                end = float(m.group(3))
                is_fall = 'fall' in act.lower()
                results.append({
                    'action': act,
                    'start': start,
                    'end': end,
                    'is_fall': is_fall
                })
            except ValueError:
                pass
    return sorted(results, key=lambda x: x['start'])


class GMDCSADataset:
    """
    Manager for the GMDCSA24 Human Fall Detection Video Dataset
    """
    def __init__(self, root_dir="ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"):
        self.root_dir = root_dir
        self.records = []
        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            return

        subjects = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) and 'Subject' in d]
        subjects.sort()

        for subj in subjects:
            subj_path = os.path.join(self.root_dir, subj)
            for cat in ['Fall', 'ADL']:
                csv_path = os.path.join(subj_path, f"{cat}.csv")
                video_dir = os.path.join(subj_path, cat)
                if not os.path.exists(csv_path):
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    # Standardize column names
                    df.columns = [c.strip() for c in df.columns]

                    for _, row in df.iterrows():
                        fname = str(row.get('File Name', '')).strip()
                        if not fname.endswith('.mp4'):
                            fname = f"{fname.zfill(2)}.mp4" if fname.isdigit() else f"{fname}.mp4"

                        vpath = os.path.join(video_dir, fname)
                        if not os.path.exists(vpath):
                            # Try finding by basename without leading 0
                            alt_matches = glob.glob(os.path.join(video_dir, f"*{os.path.splitext(fname)[0]}*.mp4"))
                            if alt_matches:
                                vpath = alt_matches[0]
                            else:
                                continue

                        classes_raw = str(row.get('Classes', ''))
                        intervals = parse_intervals(classes_raw)

                        length_val = row.get('Length (seconds)', 0)
                        try:
                            length_sec = float(length_val)
                        except (ValueError, TypeError):
                            length_sec = 0.0

                        self.records.append({
                            'subject': subj,
                            'category': cat,
                            'file_name': os.path.basename(vpath),
                            'video_path': vpath,
                            'length_sec': length_sec,
                            'time_of_recording': str(row.get('Time of Recording', 'Day')).strip(),
                            'attire': str(row.get('Attire', 'Standard')).strip(),
                            'description': str(row.get('Description', '')).strip(),
                            'classes_raw': classes_raw,
                            'intervals': intervals,
                            'fall_count': sum(1 for i in intervals if i['is_fall'])
                        })
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        self.df = pd.DataFrame(self.records)

    def get_subjects(self):
        if hasattr(self, 'df') and not self.df.empty:
            return sorted(self.df['subject'].unique().tolist())
        return ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']

    def get_categories(self):
        return ['Fall', 'ADL']

    def filter_videos(self, subject=None, category=None):
        if not hasattr(self, 'df') or self.df.empty:
            return []
        sub = self.df
        if subject:
            sub = sub[sub['subject'] == subject]
        if category:
            sub = sub[sub['category'] == category]
        return sub.to_dict('records')

    def get_video_record(self, subject, category, file_name):
        matches = [
            r for r in self.records
            if r['subject'] == subject and r['category'] == category and r['file_name'] == file_name
        ]
        return matches[0] if matches else None

    def get_dataset_stats(self):
        if not hasattr(self, 'df') or self.df.empty:
            return {"total_videos": 0, "fall_videos": 0, "adl_videos": 0}
        return {
            "total_videos": len(self.df),
            "fall_videos": len(self.df[self.df['category'] == 'Fall']),
            "adl_videos": len(self.df[self.df['category'] == 'ADL']),
            "subjects_count": self.df['subject'].nunique(),
            "avg_duration": self.df['length_sec'].mean()
        }
