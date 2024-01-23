from Preprocessing import NotesProcessing
from Params_setting import data_path


# for path in data_path:
    # processing.open_files(path)
# sequences = processing.preprocessing()

for i in range(len(data_path)):
    processing = NotesProcessing()
    processing.open_files(data_path[i])
    processing.preprocessing(data_path[i][-4:])