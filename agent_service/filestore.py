import json
import os


class FileStore:
    def __init__(self, file_path="memory_store.json"):
        self.file_path = file_path

    def save(self, key, value):
        """Save data to JSON file"""
        data = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print("Error: {e}")
                    pass

        data[key] = value

        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, key):
        """Load data from JSON file"""
        if not os.path.exists(self.file_path):
            return None
        with open(self.file_path, "r") as f:
            try:
                data = json.load(f)
                return data.get(key)
            except json.JSONDecodeError as e:
                print("Error: {e}")
                return None
