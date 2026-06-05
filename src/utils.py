import os
import dill

def save_object(file_path,obj):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    with open(file_path,"wb") as file:
        dill.dump(obj,file)

def load_object(file_path):

    with open(file_path,"rb") as file:
        return dill.load(file)