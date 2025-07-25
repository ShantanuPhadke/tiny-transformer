import requests

class FileProcessor:
    def __init__(self, file_url):
        self.file_url = file_url

    def fetch_file(self):
        response = requests.get(self.file_url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception("Failed to fetch file from URL: " + self.file_url)
    
    def save_file(self, file_path, content):
        try:
            with open(file_path, "w") as f:
                f.write(content)
        except FileExistsError:
            print(f"File {file_path} already exists.")
    
    def get_file_content(self, file_path):
        try:
            with open(file_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            raise Exception(f"File {file_path} not found.")
