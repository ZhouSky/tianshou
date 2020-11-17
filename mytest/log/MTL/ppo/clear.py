import os


if __name__ == '__main__':
    for f in os.listdir('./'):
        file_path = os.path.join('./', f)
        if os.path.isfile(file_path) and f.startswith('events'):
            print(f'delete {file_path}')
            os.remove(file_path)
